import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from Net_Capsule import CapsLayer
from Net_DPCNN import DPCNNlayer
from Transformer import Positional_Encoding, Encoder


class BagAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BagAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = encoder_outputs * weights.unsqueeze(-1)
        return outputs, weights


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
        alpha1 = F.softmax(torch.matmul(encoder_outputs, self.w1), dim=1)
        alpha2 = F.softmax(torch.matmul(encoder_outputs, self.w2), dim=1)
        alpha3 = F.softmax(torch.matmul(encoder_outputs, self.w3), dim=1)
        out1 = encoder_outputs * alpha1
        out2 = encoder_outputs * alpha2
        out3 = encoder_outputs * alpha3
        out = torch.cat([out1, out2], 2)
        out = torch.cat([out, out3], 2)
        out = torch.sum(out, 1)
        stacked_tensors = torch.stack((alpha1.squeeze(), alpha2.squeeze(), alpha3.squeeze()), dim=0)
        summed_tensors = torch.sum(stacked_tensors, dim=0)
        weights = summed_tensors / 3
        return out, weights




def _split_bags(embeddings, input_ids):
    if isinstance(input_ids, torch.Tensor):
        input_ids_t = input_ids.to(device=embeddings.device, dtype=torch.long)
    else:
        input_ids_t = torch.tensor(input_ids, device=embeddings.device, dtype=torch.long)
    unique_ids, counts = torch.unique_consecutive(input_ids_t, return_counts=True)
    splits = torch.split(embeddings, counts.tolist(), dim=0)
    return splits


class MHCpre_model_MIL_Capsule2(nn.Module):
    def __init__(self, esm_model):
        super().__init__()
        self.ESM_embedding = esm_model
        self.dropout = nn.Dropout(0.3)

        self.dpcnn = DPCNNlayer()
        self.caps_net = CapsLayer(input_caps=832, input_dim=48, output_caps=20, output_dim=10)
        self.linear = nn.Linear(200, 200)
        self.bag_attention = BagAttention_Para(200)

        self.fc_feat = nn.Sequential(
            nn.Linear(200*3, 200),
            nn.LeakyReLU(),
        )
        self.fc_out = nn.Sequential(
            self.dropout,
            nn.Linear(200, 1)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(200*3, 200),
        #     nn.LeakyReLU(),
        #     self.dropout,
        #     nn.Linear(200, 1)
        # )


    def forward(self, input_data, input_ids):
        with torch.inference_mode():
            emb_output = self.ESM_embedding(input_data, repr_layers=[30], return_contacts=False)
        embedding = emb_output["representations"][30].clone()

        x = self.dpcnn(embedding)
        x = x.view(x.size(0), 52, 16, 48)
        x = x.reshape(x.size(0), 832, 48)
        x, _ = self.caps_net(x)
        x = x.view(x.size(0), -1)
        instance_out = self.linear(x)
        instance_out = F.leaky_relu(instance_out)

        splits = _split_bags(instance_out, input_ids)
        data_feature = []
        atttn_list = []
        for bag_data in splits:
            bag_data = bag_data.unsqueeze(0)
            bag_embeddings, atttn_weights = self.bag_attention(bag_data)
            data_feature.append(bag_embeddings)
            atttn_weights = atttn_weights.unsqueeze(0)
            atttn_list.append(atttn_weights)

        atttn_list = torch.cat(atttn_list, dim=0)
        out = torch.cat(data_feature, dim=0)
        w_feat = out.cpu().detach().tolist()
        w_feat = [",".join(str(x) for x in feat_vec) for feat_vec in w_feat]
        feature = self.fc_feat(out)          # (batch, 200)
        out = self.fc_out(feature)
        # out = self.fc(out)
        out = out.squeeze()

        # # # 获取特征向量
        # w_feat = feature.cpu().detach().tolist()
        # w_feat = [",".join(str(x) for x in feat_vec) for feat_vec in w_feat]

        if out.dim() == 0:
            out = out.unsqueeze(0)
        # return out, w_feat
        return out, atttn_list, w_feat
        # return out

class MHCpre_model_MIL_Capsule2_NoESM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = 640
        self.embedding = nn.Embedding(32, self.embed, padding_idx=31, _freeze=True)
        self.dropout = nn.Dropout(0.3)

        self.dpcnn = DPCNNlayer()
        self.caps_net = CapsLayer(input_caps=832, input_dim=48, output_caps=20, output_dim=10)
        self.linear = nn.Linear(200, 200)
        self.bag_attention = BagAttention_Para(200)

        self.fc = nn.Sequential(
            nn.Linear(200*3, 200),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(200, 1)
        )

    def forward(self, input_data, input_ids):
        with torch.no_grad():
            embedding = self.embedding(input_data)
        x = self.dpcnn(embedding)
        x = x.view(x.size(0), 52, 16, 48)
        x = x.reshape(x.size(0), 832, 48)
        x, _ = self.caps_net(x)
        x = x.view(x.size(0), -1)
        instance_out = self.linear(x)
        instance_out = F.leaky_relu(instance_out)

        splits = _split_bags(instance_out, input_ids)
        data_feature = []
        for bag_data in splits:
            bag_data = bag_data.unsqueeze(0)
            bag_embeddings, _ = self.bag_attention(bag_data)
            data_feature.append(bag_embeddings)
            

        out = torch.cat(data_feature, dim=0)
        out = self.fc(out)
        out = out.squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out

class MHCpre_model_MIL_Capsule2_Transformer(nn.Module):
    def __init__(self, esm_model, device):
        super().__init__()
        self.embed = 640
        self.dropout_num=0.3
        self.pad_size = 52
        self.ESM_embedding = esm_model
        self.dropout = nn.Dropout(self.dropout_num)

        self.pos_encoding = Positional_Encoding(self.embed, self.pad_size, self.dropout_num, device)

        # Standard Transformer encoder (ablation: replaces DPCNN + CapsNet)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed, nhead=8, dim_feedforward=1280,
            dropout=self.dropout_num, batch_first=True, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.linear = nn.Linear(self.embed, 200)
        self.bag_attention = BagAttention_Para(200)
        self.fc = nn.Sequential(
            nn.Linear(200 * 3, 200),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(200, 1)
        )

    def forward(self, input_data, input_ids):
        with torch.inference_mode():
            emb_output = self.ESM_embedding(input_data, repr_layers=[30], return_contacts=False)
        embedding = emb_output["representations"][30].clone()

        x = self.pos_encoding(embedding)
        x = self.transformer_encoder(x)
        x = x[:, 0, :]

        instance_out = self.linear(x)
        instance_out = F.leaky_relu(instance_out)

        splits = _split_bags(instance_out, input_ids)
        data_feature = []
        for bag_data in splits:
            bag_data = bag_data.unsqueeze(0)
            bag_embeddings, _ = self.bag_attention(bag_data)
            data_feature.append(bag_embeddings)

        out = torch.cat(data_feature, dim=0)
        out = self.fc(out)
        out = out.squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out


class MHCpre_model_MIL_Capsule2_DNN(nn.Module):
    def __init__(self, esm_model, device):
        super().__init__()
        self.embed = 640
        self.dropout_num=0.3
        self.pad_size = 52
        self.ESM_embedding = esm_model
        self.dropout = nn.Dropout(self.dropout_num)

        self.pos_encoding = Positional_Encoding(self.embed, self.pad_size, self.dropout_num, device)

        # Positionwise MLP (replaces self-attention: processes each position independently)
        self.positionwise_mlp = nn.Sequential(
            nn.Linear(self.embed, 1280),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_num),
            nn.Linear(1280, self.embed),
        )

        self.linear = nn.Linear(self.embed, 200)
        self.bag_attention = BagAttention_Para(200)
        self.fc = nn.Sequential(
            nn.Linear(200 * 3, 200),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(200, 1)
        )

    def forward(self, input_data, input_ids):
        with torch.inference_mode():
            emb_output = self.ESM_embedding(input_data, repr_layers=[30], return_contacts=False)
        embedding = emb_output["representations"][30].clone()

        x = self.pos_encoding(embedding)
        x = self.positionwise_mlp(x)  # (batch, seq_len, 640) -> same shape
        x = x.mean(dim=1)

        instance_out = self.linear(x)
        instance_out = F.leaky_relu(instance_out)

        splits = _split_bags(instance_out, input_ids)
        data_feature = []
        
        for bag_data in splits:
            bag_data = bag_data.unsqueeze(0)
            bag_embeddings, atttn_weights = self.bag_attention(bag_data)
            data_feature.append(bag_embeddings)

        out = torch.cat(data_feature, dim=0)
        out = self.fc(out)
        out = out.squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out


class MHCpre_model_MIL_Capsule2_NoMIL(nn.Module):
    """Ablation: 移除 MIL BagAttention，用均值池化代替注意力聚合."""
    def __init__(self, esm_model):
        super().__init__()
        self.ESM_embedding = esm_model
        self.dropout = nn.Dropout(0.3)

        self.dpcnn = DPCNNlayer()
        self.caps_net = CapsLayer(input_caps=832, input_dim=48, output_caps=20, output_dim=10)
        self.linear = nn.Linear(200, 200 * 3)

        self.fc = nn.Sequential(
            nn.Linear(200 * 3, 200),
            nn.LeakyReLU(),
            self.dropout,
            nn.Linear(200, 1)
        )

    def forward(self, input_data, input_ids):
        with torch.inference_mode():
            emb_output = self.ESM_embedding(input_data, repr_layers=[30], return_contacts=False)
        embedding = emb_output["representations"][30].clone()

        x = self.dpcnn(embedding)
        x = x.view(x.size(0), 52, 16, 48)
        x = x.reshape(x.size(0), 832, 48)
        x, _ = self.caps_net(x)
        x = x.view(x.size(0), -1)
        instance_out = self.linear(x)               # (total_instances, 200*3)
        instance_out = F.leaky_relu(instance_out)

        ids = torch.tensor(input_ids, device=instance_out.device)
        n_bags = int(ids.max().item()) + 1
        result = torch.zeros(n_bags, 200 * 3, device=instance_out.device, dtype=instance_out.dtype)
        result.index_add_(0, ids, instance_out)
        counts = torch.bincount(ids).float().unsqueeze(1)
        out = result / counts

        out = self.fc(out)
        out = out.squeeze()
        if out.dim() == 0:
            out = out.unsqueeze(0)
        return out



