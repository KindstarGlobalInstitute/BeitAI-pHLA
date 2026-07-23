import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import esm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_MIL import MHC_EL_split
from mhc_model_MIL import MHCpre_model_MIL_Capsule2
from train_MIL import train_MIL

import warnings
warnings.filterwarnings("ignore")

from types import SimpleNamespace

_args = SimpleNamespace(
    data_train='./data_random/df_el_train_split0_pseudo.csv',
    data_valid='./data_random/df_el_valid_split0_pseudo.csv',
    data_test='./data_random/df_el_test_SA_pseudo.csv',
    peplen=15,
    lr=5e-5,
    epochs=15,
    batch_size=200,
    wd=1e-5,
    seed=1111,
    out_name='ablation_BeitAI_SA', 
    grad_accum=2,
    eval_steps=10,
    num_workers=10,
)


def main():
    args = _args
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    print(f'Train: {args.data_train}')
    print(f'Test:  {args.data_test}')

    # ========== ESM 模型 (t30, 640-dim) ==========
    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    model = MHCpre_model_MIL_Capsule2(emb_model)

    ckpt_path = "./model_MIL/EL_Classification_train_split0.ckpt"
    state_dict = torch.load(ckpt_path, map_location=device)
    state_dict = {k.replace('caps_net1.', 'caps_net.'): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)

    for p in emb_model.parameters():
        p.requires_grad = False

    # ========== 数据 ==========
    trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
    testset = MHC_EL_split(args.data_test, max_pep_len=args.peplen)

    model = model.to(device)

    weights = torch.tensor([20.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd, eps=1e-8
    )

    # train_data = DataLoader(trainset, batch_size=args.batch_size,
    #                         shuffle=True, num_workers=args.num_workers,
    #                         pin_memory=True, collate_fn=trainset.collate_fn,
    #                         persistent_workers=args.num_workers > 0,
    #                         prefetch_factor=2 if args.num_workers > 0 else None)
    test_data = DataLoader(testset, batch_size=args.batch_size * 2,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True, collate_fn=testset.collate_fn)

    print(f'Training samples: {len(trainset)}, batch size: {args.batch_size}')
    print(f'Device: {device}')
    start_time = time.time()

    valid_data=None
    train_data = None
    train_MIL(model, optimizer, train_data, valid_data, device, criterion,
              args.epochs, gradient_accumulation_steps=args.grad_accum,
              eval_steps=args.eval_steps, out_name=args.out_name,
              test_data=test_data)

    end_time = time.time()
    print(f'Total training time: {end_time - start_time:.2f}s')


if __name__ == "__main__":
    main()
