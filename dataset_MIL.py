import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

tok_list = {'<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10,
            'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 'H': 21, 'W': 22,
            'C': 23, 'X': 24, 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31}


class MHC_EL_split(Dataset):
    def __init__(self, root, max_pep_len=15, fold_filter=None, fold_keep=True):
        self.root = root
        self.max_pep_len = max_pep_len
        self.data = pd.read_csv(self.root, sep=",")
        self.has_fold = 'fold' in self.data.columns
        if fold_filter is not None and self.has_fold:
            if fold_keep:
                self.data = self.data[self.data['fold'] == fold_filter].reset_index(drop=True)
            else:
                self.data = self.data[self.data['fold'] != fold_filter].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        epitope = row['epitope']
        mhc = row['mhc']
        if 'label' in row:
            label = row['label']
        elif 'aff' in row:
            label = row['aff']
        else:
            label = row['mass']

        epitope_list = list(epitope)
        pad_size = self.max_pep_len
        if len(epitope) < pad_size:
            epitope_list.extend(['<pad>'] * (pad_size - len(epitope_list)))
        else:
            epitope_list = epitope_list[:pad_size]
        epitope_list.insert(0, '<cls>')
        epitope_list.append('-')

        input_mhc_ids = []
        mhc_list = mhc.split('|')
        for mhc_item in mhc_list:
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
            "mhc_str": mhc,
            "labels": label}

    def collate_fn(self, batch):
        data = []
        label = []
        index = 0
        MA_id = []
        for row in batch:
            epi_temp = row['input_epi_ids']
            mhc_temp = row['input_mhc_ids']
            label.append(row['labels'])
            for mhc in mhc_temp:
                data.append(epi_temp + mhc)
                MA_id.append(index)
            index += 1
        input_data = torch.tensor(data, dtype=torch.long)
        labels = torch.tensor(label, dtype=torch.float32)

        return {
            "input_data": input_data,
            "input_ids": torch.tensor(MA_id, dtype=torch.long),
            "labels": labels}
