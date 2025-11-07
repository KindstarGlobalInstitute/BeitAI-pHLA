# -*- coding:utf-8 -*-

import math
import torch
from torch import nn
import torch.nn.functional as func

def squash(x):
    length2 = x.pow(2).sum(dim=2) + 1e-7
    length = length2.sqrt()
    x = x * (length2 / (length2 + 1) / length).view(x.size(0), x.size(1), -1)
    return x


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, device):
        super(CapsLayer, self).__init__()
        self.input_caps = input_caps
        self.input_dim = input_dim
        self.output_caps = output_caps
        self.output_dim = output_dim
        self.weights = nn.Parameter(torch.Tensor(self.input_caps, self.input_dim, self.output_caps * self.output_dim))
        self.routing_module = AgreementRouting(self.input_caps, self.output_caps, device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, u):
        u = u.unsqueeze(2)
        # print(u.shape[0])

        # weights = self.weights.unsqueeze(2)
        # print(self.weights.shape)
        # u_predict = torch.einsum('abik, ackj -> abcij', weights, u)
        # print('u_predict size is : ', u.size())
        u_predict = u.matmul(self.weights)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim)

        s = u_predict
        # print('s size is : ', list(s.size()))
        v = self.routing_module(s)
        # v = squash(s)
        # print('v squash size is : ', list(v.size()))
        probs = v.pow(2).sqrt()
        # print('probs size is : ', list(probs.size()))
        return v, probs


class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, device, n_iterations=3):
        super().__init__()
        self.n_iterations = n_iterations
        self.b = torch.zeros((input_caps, output_caps)).to(device)

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        self.b.zero_()
        c = func.softmax(self.b, dim=1)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1)
        v = squash(s)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps))
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)
                b_batch = b_batch + (u_predict * v).sum(-1)

                c = func.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                s = (c * u_predict).sum(dim=1)
                v = squash(s)
        return v