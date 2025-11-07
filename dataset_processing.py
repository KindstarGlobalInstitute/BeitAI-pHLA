# -*- coding:utf-8 -*-

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import Dataset

tok_list = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
            'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22,
            'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31}

class MHC_EL_split(Dataset):
    def __init__(self, root, max_pep_len=15):
        self.root = root
        self.max_pep_len = max_pep_len
        self.MHC_pseudo_dict = {}
        with open('./data/MHC_pseudo.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                MHC_name, pseudo_str = line.split(" ")
                self.MHC_pseudo_dict[MHC_name.strip()] = pseudo_str.strip()

        self.allele2HLA_dict = {}
        with open('./data/allelelist', 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.strip() != "":
                    allele_name, HLA_str = line.split("\t")
                    self.allele2HLA_dict[allele_name.strip()] = HLA_str.strip()


        self.data = self.load_HLA_data(self.root)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        epitope = row['Peptide']
        mhc = row['HLA']

        # epitope_list
        epitope_list = list(epitope)
        pad_size = self.max_pep_len
        if len(epitope) < pad_size:
            epitope_list.extend(['<pad>'] * (pad_size - len(epitope_list)))
        else:
            epitope_list = epitope_list[:pad_size]
        epitope_list.insert(0, '<cls>')
        epitope_list.append('-')

        # mhc_list
        input_mhc_ids = []
        mhc = mhc.replace("HLA-A*", "HLA-A").replace("HLA-B*", "HLA-B").replace("HLA-C*", "HLA-C")
        mhc_list = mhc.split(',')
        mhc_pseudo = [self.MHC_pseudo_dict[x] for x in mhc_list]

        for mhc_item in mhc_pseudo:

            mhc_list_temp = list(mhc_item)
            mhc_id = []
            if len(mhc_list_temp) != 34:
                print("MHC pseudo length error !!")
            else:
                mhc_list_temp.append('<eos>')
                for word in mhc_list_temp:
                    mhc_id.append(tok_list.get(word, tok_list.get('<unk>')))

                input_mhc_ids.append(mhc_id)

        input_epi_ids = []
        for word in epitope_list:
            input_epi_ids.append(tok_list.get(word, tok_list.get('<unk>')))
        return {
            "input_epi_ids": input_epi_ids,
            "input_mhc_ids": input_mhc_ids,
            "epi_str": epitope,
            "mhc_str": mhc}
    def load_HLA_data(self, filepath):
        df_data = pd.read_csv(filepath, sep="\t", header=0)
        has_non_hla = (~df_data["HLA"].str.startswith("HLA-", na=False)).any()
        if has_non_hla:
            df_data['HLA'] = df_data['HLA'].map(self.allele2HLA_dict)
        return df_data
    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        data = []
        data_epi_str = []
        data_mhc_str = []
        MA_id = []
        index = 0
        for row in batch:
            epi_temp = row['input_epi_ids']
            mhc_temp = row['input_mhc_ids']
            data_epi_str.append(row['epi_str'])
            data_mhc_str.append(row['mhc_str'])

            data_temp = [epi_temp + mhc for mhc in mhc_temp]
            id_temp = [index for _ in mhc_temp]
            data += data_temp
            MA_id += id_temp
            index += 1
        input_data = torch.tensor(data)
        input_ids = MA_id

        ret = {"input_data": input_data,
               "input_ids": input_ids,
               "data_epi_str": data_epi_str,
               "data_mhc_str": data_mhc_str}
        return ret
