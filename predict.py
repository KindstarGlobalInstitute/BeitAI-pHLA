<<<<<<< HEAD
# -*- coding:utf-8 -*-

import argparse
import esm
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from docker_file.dataset_processing import MHC_EL_split
from docker_file.model import MHCpre_model_MIL_Capsule


def predict_MIL(model, data, device):
    model.eval()
    peptides = []
    mhc_str = []
    pred_batch = []
    probs_batch = []
    mhc_label = []
    with torch.no_grad():
        # for batch in data:
        for batch in tqdm(data):
            input_data = batch['input_data'].to(device)
            input_ids = batch['input_ids']
            peptides += batch['data_epi_str']
            mhc_str += batch['data_mhc_str']
            with autocast():
                output, bag_weight = model(input_data, input_ids, device)
                probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()

            preds = preds.cpu().detach().numpy()
            probs = probs.cpu().detach().numpy()

            pred_batch.append(preds)
            probs_batch.append(probs)
            mhc_label += bag_weight
    # file_out.close()
    pred_all = np.concatenate(pred_batch)
    probs_all = np.concatenate(probs_batch)

    return peptides, mhc_str, pred_all, probs_all, mhc_label


def main():
    # parser = argparse.ArgumentParser(description='Running the inference model.')
    #
    # # 添加命令行参数
    # parser.add_argument('--input', type=str, required=True, help='Path to the input data file.')
    # parser.add_argument('--output', type=str, default='./output/output.txt', help='Path to save the output results.')
    # parser.add_argument('--device', type=str, default='cpu', help='Whether to use GPU for inference, e.g. "cuda:0"')
    # parser.add_argument('--bacth_size', type=int, default=64, help='Batch size of input data')
    #
    # # 解析命令行参数
    # args = parser.parse_args()

    input_data = './test_data/input_data.txt'
    output_path = './output/output.txt'
    bacth_size = 64
    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    emb_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    model = MHCpre_model_MIL_Capsule(emb_model, device)

    model.load_state_dict(torch.load('./model_file/model_weights.ckpt'))
    model.to(device)

    dataset = MHC_EL_split(input_data, max_pep_len=15)
    dataset_load = DataLoader(dataset,
                              batch_size=bacth_size,
                              shuffle=False,
                              num_workers=10,
                              pin_memory=True,
                              collate_fn=dataset.collate_fn)
    peptides, mhc_str, res_pred, res_probs, mhc_label = predict_MIL(model, dataset_load, device)
    with open(output_path, 'w') as f:
        f.write("peptides\tallele\tpred_score\tpred_label\tbest_allele_id\n")
        for i in range(len(peptides)):
            f.write(peptides[i] + "\t" + mhc_str[i] + "\t" + str(res_probs[i]) + "\t" + str(res_pred[i]) + "\t" + str(mhc_label[i]) + "\n")

if __name__ == '__main__':
    main()
=======
# -*- coding:utf-8 -*-

import argparse
import esm
import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from docker_file.dataset_processing import MHC_EL_split
from docker_file.model import MHCpre_model_MIL_Capsule


def predict_MIL(model, data, device):
    model.eval()
    peptides = []
    mhc_str = []
    pred_batch = []
    probs_batch = []
    mhc_label = []
    with torch.no_grad():
        # for batch in data:
        for batch in tqdm(data):
            input_data = batch['input_data'].to(device)
            input_ids = batch['input_ids']
            peptides += batch['data_epi_str']
            mhc_str += batch['data_mhc_str']
            with autocast():
                output, bag_weight = model(input_data, input_ids, device)
                probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()

            preds = preds.cpu().detach().numpy()
            probs = probs.cpu().detach().numpy()

            pred_batch.append(preds)
            probs_batch.append(probs)
            mhc_label += bag_weight
    # file_out.close()
    pred_all = np.concatenate(pred_batch)
    probs_all = np.concatenate(probs_batch)

    return peptides, mhc_str, pred_all, probs_all, mhc_label


def main():
    # parser = argparse.ArgumentParser(description='Running the inference model.')
    #
    # # 添加命令行参数
    # parser.add_argument('--input', type=str, required=True, help='Path to the input data file.')
    # parser.add_argument('--output', type=str, default='./output/output.txt', help='Path to save the output results.')
    # parser.add_argument('--device', type=str, default='cpu', help='Whether to use GPU for inference, e.g. "cuda:0"')
    # parser.add_argument('--bacth_size', type=int, default=64, help='Batch size of input data')
    #
    # # 解析命令行参数
    # args = parser.parse_args()

    input_data = './test_data/input_data.txt'
    output_path = './output/output.txt'
    bacth_size = 64
    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    emb_model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    model = MHCpre_model_MIL_Capsule(emb_model, device)

    model.load_state_dict(torch.load('./model_file/model_weights.ckpt'))
    model.to(device)

    dataset = MHC_EL_split(input_data, max_pep_len=15)
    dataset_load = DataLoader(dataset,
                              batch_size=bacth_size,
                              shuffle=False,
                              num_workers=10,
                              pin_memory=True,
                              collate_fn=dataset.collate_fn)
    peptides, mhc_str, res_pred, res_probs, mhc_label = predict_MIL(model, dataset_load, device)
    with open(output_path, 'w') as f:
        f.write("peptides\tallele\tpred_score\tpred_label\tbest_allele_id\n")
        for i in range(len(peptides)):
            f.write(peptides[i] + "\t" + mhc_str[i] + "\t" + str(res_probs[i]) + "\t" + str(res_pred[i]) + "\t" + str(mhc_label[i]) + "\n")

if __name__ == '__main__':
    main()
>>>>>>> b97aa2f (Initailization)
