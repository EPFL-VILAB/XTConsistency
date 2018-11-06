

import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from models import TrainableModel
from utils import *


class ConvBlock(nn.Module):
    def __init__(self, f1, f2, kernel_size=3, padding=1, use_groupnorm=True, groups=8, dilation=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (kernel_size, kernel_size), dilation=dilation, padding=padding*dilation)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(
                f1, f1, (3, 3), dilation=dilation, stride=2, padding=dilation, output_padding=1
            )
        if use_groupnorm:
            self.bn = nn.GroupNorm(groups, f1)
        else:
            self.bn = nn.BatchNorm2d(f1)

    def forward(self, x):
        # x = F.dropout(x, 0.04, self.training)
        x = self.bn(x)
        if self.transpose:
            # x = F.upsample(x, scale_factor=2, mode='bilinear')
            x = F.relu(self.convt(x))
            # x = x[:, :, :-1, :-1]
        x = F.relu(self.conv(x))
        return x


class DenseNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 96, groups=3), 
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class Dense1by1Net(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 64, groups=3, kernel_size=1, padding=0), 
            ConvBlock(64, 96, kernel_size=1, padding=0), 
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class Dense1by1end(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 64, groups=3, kernel_size=1, padding=0), 
            ConvBlock(64, 96, kernel_size=1, padding=0), 
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 1),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class DenseKernelsNet(TrainableModel):
    def __init__(self, kernel_size=7):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 64, groups=3, kernel_size=1, padding=0), 
            ConvBlock(64, 96, kernel_size=1, padding=0), 
            ConvBlock(96, 96, kernel_size=1, padding=0),
            ConvBlock(96, 96, kernel_size=kernel_size, padding=kernel_size//2),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 96),
            ConvBlock(96, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class DeepNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 32, groups=3), 
            ConvBlock(32, 32),
            ConvBlock(32, 32, dilation=2),
            ConvBlock(32, 32, dilation=2),
            ConvBlock(32, 32, dilation=4),
            ConvBlock(32, 32, dilation=4),
            ConvBlock(32, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class WideNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 32, groups=3), 
            ConvBlock(32, 32, kernel_size=5, padding=2),
            ConvBlock(32, 32, kernel_size=5, padding=2),
            ConvBlock(32, 32, kernel_size=5, padding=2),
            ConvBlock(32, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class PyramidNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 16, groups=3), 
            ConvBlock(16, 32, kernel_size=5, padding=2),
            ConvBlock(32, 64, kernel_size=5, padding=2),
            ConvBlock(64, 96, kernel_size=3, padding=1),
            ConvBlock(96, 32, kernel_size=3, padding=1),
            ConvBlock(32, 3),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)



class BaseNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            ConvBlock(3, 32, use_groupnorm=False), 
            ConvBlock(32, 32, use_groupnorm=False),
            ConvBlock(32, 32, use_groupnorm=False),
            ConvBlock(32, 1, use_groupnorm=False),
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class ResidualsNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, use_groupnorm=False), 
            ConvBlock(32, 32, use_groupnorm=False),
        )
        self.mid = nn.Sequential(
            ConvBlock(32, 64, use_groupnorm=False), 
            ConvBlock(64, 64, use_groupnorm=False),
            ConvBlock(64, 32, use_groupnorm=False),
        )
        self.decoder = nn.Sequential(
            ConvBlock(64, 32, use_groupnorm=False), 
            ConvBlock(32, 3, use_groupnorm=False),
        )

    def forward(self, x):
        tmp = self.encoder(x)
        x = F.max_pool2d(tmp, 2)
        x = self.mid(x)
        x = F.upsample(x, scale_factor=2, mode='bilinear')
        x = torch.cat([x, tmp], dim=1)
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


