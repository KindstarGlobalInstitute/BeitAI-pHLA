# -*- coding:utf-8 -*-

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from Net_Capsule import CapsLayer

class BagAttention_Para(nn.Module):
    def __init__(self, hidden_size):
        super(BagAttention_Para, self).__init__()
        self.w1 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.w1.data)
        self.w2 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.w2.data)
        self.w3 = nn.Parameter(torch.empty(hidden_size, 1))
        nn.init.xavier_uniform_(self.w3.data)

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch_size, bag_size, hidden_size]
        alpha1 = F.softmax(torch.matmul(encoder_outputs, self.w1), dim=1)  # [1, bag_num, 1]
        alpha2 = F.softmax(torch.matmul(encoder_outputs, self.w2), dim=1)  # [1, bag_num, 1]
        alpha3 = F.softmax(torch.matmul(encoder_outputs, self.w3), dim=1)  # [1, bag_num, 1]
        out1 = encoder_outputs * alpha1
        out2 = encoder_outputs * alpha2
        out3 = encoder_outputs * alpha3
        out = torch.cat([out1, out2], 2)  # [1, bag_num, 1280]
        out = torch.cat([out, out3], 2)
        out = torch.sum(out, 1)
        stacked_tensors = torch.stack((alpha1.squeeze(), alpha2.squeeze(), alpha3.squeeze()), dim=0)
        summed_tensors = torch.sum(stacked_tensors, dim=0)
        max_value, max_index = summed_tensors.max(dim=0)
        weights = max_index.item()
        return out, weights

class MHCpre_model_MIL_Capsule(nn.Module):
    def __init__(self, esm_model, device):
        super().__init__()
        self.filter_sizes1 = [1, 3, 5]
        self.filter_sizes2 = 2
        self.filter_sizes_single = 1
        self.num_kernel = 256
        self.embed = 1280
        self.device = device
        self.head_num = 4
        self.ESM_embedding = esm_model
        self.dropout = nn.Dropout(0.3)

        self.dpcnn = DPCNNlayer()
        self.caps_net1 = CapsLayer(input_caps=832, input_dim=48, output_caps=20, output_dim=10, device=self.device)
        self.linear = nn.Linear(200, 200)
        self.bag_attention = BagAttention_Para(200)

        self.fc = nn.Sequential(
            nn.Linear(200*3, 200),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(200, 1)
        )

    def conv_and_norm(self, x, conv, norm):
        x = conv(x)
        x = norm(x)
        return x

    def forward(self, input_data, input_ids, device):
        with torch.no_grad():
            emb_output = self.ESM_embedding(input_data, repr_layers=[33], return_contacts=True)
        embedding = emb_output["representations"][33]

        x = self.dpcnn(embedding)
        x = x.view(x.size(0), 52, 16, 48)
        x = x.reshape(x.size(0), 832, 48)
        x, _ = self.caps_net1(x)
        x = x.view(x.size(0), -1)
        instance_out = self.linear(x)                                   # (bag_num, 200)
        instance_out = F.leaky_relu(instance_out)

        data_feature = []
        bag_weight_list = []
        id_temp = -1
        start_id = 0
        for MA_index in range(len(input_ids)):
            if input_ids[MA_index] != id_temp:
                end_id = MA_index
                if end_id != 0:
                    data_bag = instance_out[start_id:end_id, :]
                    data_bag = data_bag.unsqueeze(0)  # (1, bag_num, 200)
                    bag_embeddings, bag_weight = self.bag_attention(data_bag)  # (1, 200 * 4 )
                    data_feature.append(bag_embeddings)
                    bag_weight_list.append(bag_weight)

                start_id = MA_index
                id_temp = input_ids[MA_index]
            if MA_index == len(input_ids) - 1:
                data_bag = instance_out[start_id:, :]
                data_bag = data_bag.unsqueeze(0)  # (1, bag_num, 200)
                bag_embeddings, bag_weight = self.bag_attention(data_bag)  # (1, 200 * 3)
                data_feature.append(bag_embeddings)
                bag_weight_list.append(bag_weight)
        out = torch.cat(data_feature, dim=0)  # (128, 200 * 3)
        out = self.fc(out)
        out = out.squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out, bag_weight_list


class DPCNNlayer(nn.Module):
    def __init__(self):
        super(DPCNNlayer, self).__init__()
        self.embed = 1280
        self.layer_num = 3

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 256, (1, self.embed), bias=False),
            nn.BatchNorm2d(256))
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(1, 256, (1, self.embed), bias=False),
            nn.BatchNorm2d(256))

        self.bottleneck1 = nn.Sequential(
            nn.Conv2d(256, 64, (1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256)
        )

        self.shortcut2 = nn.Sequential(
            nn.Conv2d(256, 256 * 3, (1, 1), bias=False),
            nn.BatchNorm2d(256 * 3))

        self.bottleneck2 = nn.Sequential(
            nn.Conv2d(256 * 3, 256, (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, (5, 1), padding=(2, 0), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256 * 3, (1, 1), bias=False),
            nn.BatchNorm2d(256 * 3)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch_size, 250, seq_len, 1]
        x = self.conv1(x)
        px = x
        for _ in range(self.layer_num):
            x = self.bottleneck1(x)
        out = px + x                 # [batch_size, 256, seq_len, 1]
        out = self.shortcut2(out)
        px2 = out
        for _ in range(self.layer_num):
            out = self.bottleneck2(out)
        out = px2 + out                # [batch_size, 256*4, seq_len, 1]
        out = out.squeeze(-1)  # [batch_size, num_filters(250)]
        out = out.permute(0, 2, 1)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # x: [seq_len, batch_size, input_size]
        x = self.embedding(x)  # [seq_len, batch_size, hidden_size]
        x = self.transformer_encoder(x)  # [seq_len, batch_size, hidden_size]
        x = x[0, :, :]
        return x  # [batch_size, hidden_size]