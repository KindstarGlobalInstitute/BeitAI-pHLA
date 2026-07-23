import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import esm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from dataset_MIL import MHC_EL_split
from mhc_model_MIL import MHCpre_model_MIL_Capsule2
from train_MIL import train_MIL

from types import SimpleNamespace
import warnings
warnings.filterwarnings("ignore")

_base_args = SimpleNamespace(
    data_train='./data_random/df_el_train_split0_pseudo.csv',
    data_valid='./data_random/df_el_valid_split0_pseudo.csv',
    data_test='./data_random/df_el_test_pseudo.csv',
    peplen=15,
    lr=5e-5,
    epochs=15,
    batch_size=256,
    wd=1e-5,
    seed=1111,
    out_name='sens',
    grad_accum=1,
    eval_steps=10,
    num_workers=4,
    gpu_ids='0,1',
    master_port='12356',
)

pos_weights_to_test = [5, 7, 10, 15, 20, 25]

def train_worker(rank, world_size, args, pos_weight):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port

    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    if rank == 0:
        print(f'World size: {world_size}, pos_weight: {pos_weight}')

    trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
    validset = MHC_EL_split(args.data_valid, max_pep_len=args.peplen)
    testset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)

    if rank == 0:
        pos_train = int(trainset.data['mass'].sum())
        neg_train = len(trainset.data) - pos_train
        pos_test = int(testset.data['mass'].sum())
        neg_test = len(testset.data) - pos_test
        print(f'Train: {len(trainset)} samples, pos={pos_train}, neg={neg_train}, ratio={neg_train/pos_train:.1f}:1')
        print(f'Test: {len(testset)} samples, pos={pos_test}, neg={neg_test}, ratio={neg_test/pos_test:.1f}:1')

    train_sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(validset, num_replicas=world_size, rank=rank, shuffle=False)

    train_data = DataLoader(trainset, batch_size=args.batch_size,
                            sampler=train_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=trainset.collate_fn,
                            persistent_workers=args.num_workers > 0,
                            prefetch_factor=2 if args.num_workers > 0 else None)
    valid_data = DataLoader(validset, batch_size=args.batch_size * 2,
                            sampler=valid_sampler,
                            num_workers=args.num_workers, pin_memory=True,
                            collate_fn=validset.collate_fn)
    test_data = DataLoader(testset, batch_size=args.batch_size * 2,
                           shuffle=False,
                           num_workers=args.num_workers, pin_memory=True,
                           collate_fn=testset.collate_fn)

    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    model = MHCpre_model_MIL_Capsule2(emb_model)
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    weights = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=args.wd, eps=1e-8)

    out_name = f"{args.out_name}_pw{pos_weight}"

    train_MIL(model, optimizer, train_data, valid_data, device, criterion,
              args.epochs, gradient_accumulation_steps=args.grad_accum,
              eval_steps=args.eval_steps, out_name=out_name,
              test_data=test_data)

    dist.destroy_process_group()


def run_sensitivity(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPU(s) for sensitivity analysis')
    os.makedirs("sensitivity_output", exist_ok=True)

    results = []
    metrics_names = None

    for pw in pos_weights_to_test:
        start_time = time.time()
        mp.spawn(train_worker, args=(num_gpus, args, pw), nprocs=num_gpus, join=True)
        elapsed = time.time() - start_time
        out_name = f"{args.out_name}_pw{pw}"
        test_result_path = f"./output_MIL/loss/test_{out_name}.txt"
        if os.path.exists(test_result_path):
            with open(test_result_path) as f:
                header = f.readline().strip()
                values = f.readline().strip()
            if metrics_names is None:
                metrics_names = header.split('\t')
            result = dict(zip(metrics_names, [float(v) for v in values.split('\t')]))
            result['pos_weight'] = pw
            result['time_sec'] = elapsed
            results.append(result)
            print(f'pos_weight={pw:>4.1f}: AUPRC={result["auprc"]:.4f}, F1={result["f1"]:.4f}, '
                  f'P={result["pre"]:.4f}, R={result["recall"]:.4f}, Time={elapsed:.1f}s')
        else:
            print(f'Warning: test result not found for pos_weight={pw}')

    if not results:
        print('No results collected.')
        return

    summary_path = "sensitivity_output/pos_weight_summary.txt"
    metrics = ['pos_weight', 'time_sec'] + metrics_names
    with open(summary_path, "w") as f:
        f.write("\t".join(metrics) + "\n")
        for r in results:
            f.write("\t".join([f"{r.get(m, 0):.4f}" for m in metrics]) + "\n")


if __name__ == "__main__":
    run_sensitivity(_base_args)
