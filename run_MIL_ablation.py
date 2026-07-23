import os
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import esm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from dataset_MIL import MHC_EL_split
from mhc_model_MIL import MHCpre_model_MIL_Capsule2_DNN
from train_MIL import train_MIL
import shutil
import warnings
warnings.filterwarnings("ignore")

from types import SimpleNamespace

_args = SimpleNamespace(
    data_train='./data_random/df_el_train_split0_pseudo.csv',
    data_valid='./data_random/df_el_valid_split0_pseudo.csv',
    data_test='./data_random/df_el_test_MA_pseudo.csv',
    peplen=15,
    lr=5e-5,
    epochs=15,
    batch_size=200,
    wd=1e-5,
    seed=1111,
    out_name='ablation_DNN_SA_data',
    grad_accum=2,
    eval_steps=10,
    num_workers=10,
)

def get_pseudo_dict():
    pseudo_dat = './data_random/MHC_pseudo.dat'
    MHC_pseudo_dict = {}
    with open(pseudo_dat) as f:
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) >= 2:
                MHC_pseudo_dict[parts[0]] = parts[1]
    return MHC_pseudo_dict

def main():
    args = _args
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)

    print('=== Ablation: Transformer (replace DPCNN + CapsNet) ===')
    print(f'Train: {args.data_train}')
    print(f'Test:  {args.data_test}')
    MHC_pseudo_dict = get_pseudo_dict()

    # ========== ESM 模型 (t30, 640-dim) ==========
    emb_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    model = MHCpre_model_MIL_Capsule2_DNN(emb_model,device=device)

    ckpt_path = "./model_MIL/EL_Classification_ablation_DNN_split0.ckpt"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    for p in emb_model.parameters():
        p.requires_grad = False
    model = model.to(device)

    weights = torch.tensor([20.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.wd, eps=1e-8
    )

    MA_test_path = "./data_random/SA_test/HLA_seen/"
    is_pseudo_file = False
    temp_dir = '/tmp/SA_predict/'
    os.makedirs(temp_dir, exist_ok=True)

    for file in os.listdir(MA_test_path):
        data_test = MA_test_path + file
        if not is_pseudo_file:
            hla_tmp = file.replace('.txt', '')
            key = hla_tmp.replace(':', '').replace('*', '')
            pseudo_str = MHC_pseudo_dict.get(key, hla_tmp)
            df = pd.read_csv(data_test, sep='\t', header=None, names=['epitope', 'mass', 'mhc'])
            df_temp = df[['epitope', 'mass']].copy()
            df_temp['mhc'] = pseudo_str
            temp_path = os.path.join(temp_dir, file.replace('.txt', '.csv'))
            df_temp.to_csv(temp_path, index=False)
            data_test = temp_path

        trainset = MHC_EL_split(args.data_train, max_pep_len=args.peplen)
        testset = MHC_EL_split(data_test, max_pep_len=args.peplen)
        # train_data = DataLoader(trainset, batch_size=args.batch_size,
        #                         shuffle=True, num_workers=args.num_workers,
        #                         pin_memory=True, collate_fn=trainset.collate_fn,
        #                         persistent_workers=args.num_workers > 0,
        #                         prefetch_factor=2 if args.num_workers > 0 else None)
        test_data = DataLoader(testset, batch_size=args.batch_size * 2,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True, collate_fn=testset.collate_fn)

        print(f'Testing samples: {len(testset)}, batch size: {args.batch_size * 2}')
        print(f'Device: {device}')

        valid_data=None
        train_data = None
        train_MIL(model, optimizer, train_data, valid_data, device, criterion,
                args.epochs, gradient_accumulation_steps=args.grad_accum,
                eval_steps=args.eval_steps, out_name=args.out_name,
                test_data=test_data)
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
