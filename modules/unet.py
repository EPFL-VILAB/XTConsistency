
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from models import TrainableModel
from utils import *


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)        
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, prev_feature_map, x):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):

        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        if self.down_size:
            x = self.max_pool(x)
        return x


class UNet(TrainableModel):
    def __init__(self):
        super().__init__()

        self.down_block1 = UNet_down_block(3, 16, False) #   256
        self.down_block2 = UNet_down_block(16, 32, True) #   128
        self.down_block3 = UNet_down_block(32, 64, True) #   64
        self.down_block4 = UNet_down_block(64, 128, True) #  32
        self.down_block5 = UNet_down_block(128, 256, True) # 16
        self.down_block6 = UNet_down_block(256, 512, True) # 8
        self.down_block7 = UNet_down_block(512, 1024, True)# 4 

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, 1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, 1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, 1024)

        self.up_block1 = UNet_up_block(512, 1024, 512)
        self.up_block2 = UNet_up_block(256, 512, 256)
        self.up_block3 = UNet_up_block(128, 256, 128)
        self.up_block4 = UNet_up_block(64, 128, 64)
        self.up_block5 = UNet_up_block(32, 64, 32)
        self.up_block6 = UNet_up_block(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, 3, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)
        self.x7 = self.down_block7(self.x6)

        self.x7 = self.relu(self.bn1(self.mid_conv1(self.x7)))
        self.x7 = self.relu(self.bn2(self.mid_conv2(self.x7)))
        self.x7 = self.relu(self.bn3(self.mid_conv3(self.x7)))

        x = self.up_block1(self.x6, self.x7)
        x = self.up_block2(self.x5, x)
        x = self.up_block3(self.x4, x)
        x = self.up_block4(self.x3, x)
        x = self.up_block5(self.x2, x)
        x = self.up_block6(self.x1, x)
        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)