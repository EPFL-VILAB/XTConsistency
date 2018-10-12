
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
    def __init__(self, f1, f2, use_groupnorm=True, groups=8, dilation=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (3, 3), dilation=dilation, padding=dilation)
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


class ResNet(TrainableModel):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet50()
        self.final_conv = nn.Conv2d(2048, 8, (3, 3), padding=1)

        self.decoder = nn.Sequential(
            ConvBlock(8, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128),
            ConvBlock(128, 128, transpose=True),
            ConvBlock(128, 128, transpose=True),
            ConvBlock(128, 128, transpose=True),
            ConvBlock(128, 128, transpose=True),
            ConvBlock(128, 3, transpose=True),
        )

    def forward(self, x):

        for layer in list(self.resnet._modules.values())[:-2]:
            x = layer(x)

        x = self.final_conv(x)
        x = self.decoder(x)

        return x

    ### Not in Use Right Now ###
    def loss(self, pred, target):
        mask = build_mask(pred, val=0.502)
        mse = F.mse_loss(pred[mask], target[mask])
        return mse, (mse.detach(),)
