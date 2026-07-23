import torch
import torch.nn as nn


class DPCNNlayer(nn.Module):
    def __init__(self):
        super(DPCNNlayer, self).__init__()
        self.embed = 640
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
        x = x.unsqueeze(1)
        x = self.conv1(x)
        px = x
        for _ in range(self.layer_num):
            x = self.bottleneck1(x)
        out = px + x
        out = self.shortcut2(out)
        px2 = out
        for _ in range(self.layer_num):
            out = self.bottleneck2(out)
        out = px2 + out
        out = out.squeeze(-1)
        out = out.permute(0, 2, 1)
        return out
