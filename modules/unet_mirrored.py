
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from models import TrainableModel
from utils import *


class UNet_up_block(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super().__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3)
        self.bn3 = nn.GroupNorm(8, output_channel)        
        self.relu = torch.nn.ReLU()
        self.up_sample = up_sample

    def forward(self, prev_feature_map, x):
        if self.up_sample:
            x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(F.pad(self.conv1(x), (1, 1, 1, 1), mode='reflect')))
        x = self.relu(self.bn2(F.pad(self.conv2(x), (1, 1, 1, 1), mode='reflect')))
        x = self.relu(self.bn3(F.pad(self.conv3(x), (1, 1, 1, 1), mode='reflect')))
        return x


class UNet_down_block(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3)
        self.bn1 = nn.GroupNorm(8, output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3)
        self.bn2 = nn.GroupNorm(8, output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3)
        self.bn3 = nn.GroupNorm(8, output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        x = self.relu(self.bn1(F.pad(self.conv1(x), (1, 1, 1, 1), mode='reflect')))
        x = self.relu(self.bn2(F.pad(self.conv2(x), (1, 1, 1, 1), mode='reflect')))
        x = self.relu(self.bn3(F.pad(self.conv3(x), (1, 1, 1, 1), mode='reflect')))
        if self.down_size:
            x = self.max_pool(x)
        return x


class UNet(TrainableModel):
    def __init__(self, downsample=6, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        pad = [False for i in range(0, self.downsample)]
        for i in range(0, self.downsample):
            if x.shape[2] % 2 != 0: 
                x = F.pad(x, (1, 0, 1, 0))
                pad[i] = True
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            # print (x.shape, xvals[i].shape)
            x = self.up_blocks[i](xvals[i], x)
            if pad[i] != 0: 
                x = x[:, :, 1:, 1:]

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

class UNetReshade(TrainableModel):
    def __init__(self,  downsample=6, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels, self.downsample = in_channels, out_channels, downsample
        self.down1 = UNet_down_block(in_channels, 16, False)
        self.down_blocks = nn.ModuleList(
            [UNet_down_block(2**(4+i), 2**(5+i), True) for i in range(0, downsample)]
        )

        bottleneck = 2**(4 + downsample)
        self.mid_conv1 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, bottleneck)
        self.mid_conv2 = nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn2 = nn.GroupNorm(8, bottleneck)
        self.mid_conv3 = torch.nn.Conv2d(bottleneck, bottleneck, 3, padding=1)
        self.bn3 = nn.GroupNorm(8, bottleneck)

        self.up_blocks = nn.ModuleList(
            [UNet_up_block(2**(4+i), 2**(5+i), 2**(4+i)) for i in range(0, downsample)]
        )

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.GroupNorm(8, 16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.down1(x)
        xvals = [x]
        for i in range(0, self.downsample):
            x = self.down_blocks[i](x)
            xvals.append(x)

        x = self.relu(self.bn1(self.mid_conv1(x)))
        x = self.relu(self.bn2(self.mid_conv2(x)))
        x = self.relu(self.bn3(self.mid_conv3(x)))

        for i in range(0, self.downsample)[::-1]:
            x = self.up_blocks[i](xvals[i], x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.relu(self.last_conv2(x))
        x = x.clamp(max=1, min=0).mean(dim=1, keepdim=True)
        x = x.expand(-1, 3, -1, -1)
        return x

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


class UNetOld(TrainableModel):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.down_block1 = UNet_down_block(in_channels, 16, False) #   256
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
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
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

class UNetOld2(TrainableModel):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        self.in_channels, self.out_channels = in_channels, out_channels
        self.initial = nn.Sequential(
            ConvBlock(in_channels, 16, groups=3, kernel_size=1, padding=0),
            ConvBlock(16, 16, groups=4, kernel_size=1, padding=0)
        )
        self.down_block1 = UNet_down_block(16, 16, False)
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
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.initial(x)
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


if __name__ == "__main__":

    model = UNet()
    x = torch.randn(1, 3, 256, 256)
    print (model(x).shape)

        
