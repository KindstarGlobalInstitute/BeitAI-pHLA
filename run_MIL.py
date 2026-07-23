import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import esm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from dataset_MIL import MHC_EL_split
from mhc_model_MIL import MHCpre_model_MIL_Capsule2
from train_MIL import train_MIL

import warnings
warnings.filterwarnings("ignore")

from types import SimpleNamespace

_args = SimpleNamespace(
    data_train='./data_random/df_el_train_split0_pseudo.csv',
    data_valid='./data_random/df_el_valid_split0_pseudo.csv',
    data_test='./data_random/df_el_test_pseudo.csv',
    peplen=15,
    lr=5e-5,
    epochs=15,
    batch_size=256,
    wd=1e-5,
    seed=1111,
    out_name='train_0626',
    grad_accum=2,
    eval_steps=10,
    num_workers=20,
    ddp=True,
    master_port='12355',
    gpu_ids='0,1,3',
    valid_fold=None,
    cv_folds=None,
    split_mode=True,
    n_splits=3,
    load_ckpt=None,  # 续训练时设为之前 out_name 前缀，如 'train_0608'
)


def train_worker(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port


    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=world_size, rank=rank)

    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    if rank == 0:
        print(f'World size: {world_size}, using DDP with GPUs 0-{world_size - 1}')

    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    model = MHCpre_model_MIL_Capsule2(emb_model)
    for p in emb_model.parameters():
        p.requires_grad = False

    if args.split_mode:
        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
        validset = MHC_EL_split(args.data_valid, max_pep_len=args.peplen)
        # testset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)
        testset = None
    elif args.valid_fold is not None:
        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen, fold_filter=args.valid_fold, fold_keep=False)
        validset = MHC_EL_split(args.data_train, max_pep_len=args.peplen, fold_filter=args.valid_fold, fold_keep=True)
        testset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)
    else:
        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
        validset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)
        testset = None

    model = model.to(device)
    if args.resume_ckpt is not None:
        if rank == 0:
            print(f"Loading checkpoint: {args.resume_ckpt}")
        if os.path.exists(args.resume_ckpt):
            model.load_state_dict(torch.load(args.resume_ckpt, map_location=device))
        elif rank == 0:
            print(f"Warning: checkpoint not found: {args.resume_ckpt}")

    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    weights = torch.tensor([20.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=args.wd, eps=1e-8)
    train_sampler = DistributedSampler(trainset, num_replicas=world_size,rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(validset, num_replicas=world_size,rank=rank, shuffle=False)
    train_data = DataLoader(trainset, batch_size=args.batch_size,
                            sampler=train_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=trainset.collate_fn,
                            persistent_workers=args.num_workers > 0,
                            prefetch_factor=2 if args.num_workers > 0 else None)
    valid_data = DataLoader(validset, batch_size=args.batch_size * 2,
                            sampler=valid_sampler,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=validset.collate_fn)
    if rank == 0:
        total = len(trainset)
        print(f'Training samples: {total}, per GPU: {total // world_size}')
        print(f'Batch size per GPU: {args.batch_size}, total: {args.batch_size * world_size}')

    test_data = None
    if testset is not None:
        test_data = DataLoader(testset, batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=testset.collate_fn)

    train_MIL(model, optimizer, train_data, valid_data, device, criterion,
              args.epochs, gradient_accumulation_steps=args.grad_accum,
              eval_steps=args.eval_steps, out_name=args.out_name,
              test_data=test_data)

    dist.destroy_process_group()


def train_single_gpu(args):
    device = torch.device('cuda:0')
    torch.manual_seed(args.seed)

    print('Running single-GPU training (DDP disabled)')

    # ========== 在线 ESM 路径 ==========
    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    model = MHCpre_model_MIL_Capsule2(emb_model)
    print("加载完毕")
    for p in emb_model.parameters():
        p.requires_grad = False

    if args.split_mode:
        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
        validset = MHC_EL_split(args.data_valid, max_pep_len=args.peplen)
        testset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)
    elif args.valid_fold is not None:
        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen,
                                fold_filter=args.valid_fold, fold_keep=False)
        validset = MHC_EL_split(args.data_train, max_pep_len=args.peplen,
                                fold_filter=args.valid_fold, fold_keep=True)
        testset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)
    else:
        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
        validset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)
        testset = None

    model = model.to(device)

    if args.resume_ckpt is not None:
        print(f"Loading checkpoint: {args.resume_ckpt}")
        if os.path.exists(args.resume_ckpt):
            model.load_state_dict(torch.load(args.resume_ckpt, map_location=device))
        else:
            print(f"Warning: checkpoint not found: {args.resume_ckpt}")

    weights = torch.tensor([7.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=args.lr, weight_decay=args.wd, eps=1e-8)

    train_data = DataLoader(trainset, batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=trainset.collate_fn,
                            persistent_workers=args.num_workers > 0,
                            prefetch_factor=2 if args.num_workers > 0 else None)

    valid_data = DataLoader(validset, batch_size=args.batch_size * 2,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            collate_fn=validset.collate_fn)

    print(f'Training samples: {len(trainset)}, batch size: {args.batch_size}')
    start_time = time.time()

    test_data = None
    if testset is not None:
        test_data = DataLoader(testset, batch_size=args.batch_size * 2,
                               shuffle=False, num_workers=args.num_workers,
                               pin_memory=True, collate_fn=testset.collate_fn)

    train_MIL(model, optimizer, train_data, valid_data, device, criterion,
              args.epochs, gradient_accumulation_steps=args.grad_accum,
              eval_steps=args.eval_steps, out_name=args.out_name,
              test_data=test_data)

    end_time = time.time()
    print(f'Total training time: {end_time - start_time:.2f}s')


if __name__ == "__main__":
    args = _args

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    num_gpus = torch.cuda.device_count()
    print(f'Detected {num_gpus} GPU(s)')

    use_ddp = args.ddp and num_gpus > 1

    base_out_name = args.out_name
    all_test_results = []

    if args.split_mode:
        iterations = list(range(args.n_splits))
    elif args.cv_folds is not None:
        iterations = list(range(args.cv_folds))
    else:
        iterations = [args.valid_fold]
    
    for iter_idx, iter_val in enumerate(iterations):

        if args.split_mode:
            # 随机拆分模式：更新到对应拆分的数据文件
            split = iter_val
            args.data_train = f'./data_random/df_el_train_split{split}_pseudo.csv'
            args.data_valid = f'./data_random/df_el_valid_split{split}_pseudo.csv'
            args.out_name = f"{base_out_name}_split{split}"
            label = f'Split {split}'
        elif args.cv_folds is not None:
            args.valid_fold = iter_val
            args.out_name = f"{base_out_name}_fold{iter_val}"
            label = f'Fold {iter_val}'
        else:
            args.valid_fold = iter_val
            args.out_name = base_out_name
            label = f'Training (valid_fold={iter_val})'

        print(f'\n{"=" * 60}')
        print(f'{label} ({iter_idx + 1}/{len(iterations)})')
        print(f'Train: {args.data_train}')
        if not args.split_mode:
            print(f'Valid: from data_train (fold filter)')
        else:
            print(f'Valid: {args.data_valid}')
        print(f'Test:  {args.data_test}')
        print(f'Output: {args.out_name}')
        print(f'{"=" * 60}')

        if args.load_ckpt is not None:
            args.resume_ckpt = f"./model_MIL/EL_Classification_{args.load_ckpt}_split{iter_val}.ckpt"
            print(f'Resume from checkpoint: {args.resume_ckpt}')
        else:
            args.resume_ckpt = None

        if use_ddp:
            mp.spawn(train_worker, args=(num_gpus, args), nprocs=num_gpus, join=True)
        else:
            train_single_gpu(args)

        test_result_path = f"./output_MIL/loss/test_{args.out_name}.txt"
        if os.path.exists(test_result_path):
            with open(test_result_path) as f:
                header = f.readline().strip()
                values = f.readline().strip()
            metrics_names = header.split('\t')
            result = dict(zip(metrics_names, [float(v) for v in values.split('\t')]))
            all_test_results.append(result)
            print(f'{label} test: AUPRC={result["auprc"]:.4f}, F1={result["f1"]:.4f}, '
                  f'Precision={result["pre"]:.4f}, Recall={result["recall"]:.4f}')
        else:
            print(f'Warning: test result file not found: {test_result_path}')

    args.out_name = base_out_name

    # CV 汇总
    if len(all_test_results) > 1:
        print(f'\n{"=" * 60}')
        title = 'Random Splits Summary' if args.split_mode else 'Cross-Validation Summary'
        print(f'{title} ({len(iterations)} runs)')
        print(f'{"=" * 60}')
        metrics = list(all_test_results[0].keys())
        for metric in metrics:
            values = [r[metric] for r in all_test_results]
            mean = np.mean(values)
            std = np.std(values)
            print(f'{metric}: {mean:.4f} ± {std:.4f}')

        os.makedirs("output_MIL", exist_ok=True)
        summary_path = f"./output_MIL/loss/test_{base_out_name}_summary.txt"
        with open(summary_path, "w") as f:
            id_header = "split" if args.split_mode else "fold"
            f.write(id_header + "\t" + "\t".join(metrics) + "\n")
            for i, val in enumerate(iterations):
                f.write(f"{val}\t" + "\t".join([f"{all_test_results[i][m]:.4f}" for m in metrics]) + "\n")
            f.write("mean\t" + "\t".join([f"{np.mean([r[m] for r in all_test_results]):.4f}" for m in metrics]) + "\n")
            f.write("std\t" + "\t".join([f"{np.std([r[m] for r in all_test_results]):.4f}" for m in metrics]) + "\n")
        print(f'\nSummary saved to {summary_path}')
    elif len(all_test_results) == 1:
        print(f'\nSingle run complete. AUPRC={all_test_results[0]["auprc"]:.4f}')
