
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from models import TrainableModel
from utils import *

class ResNetDisc(TrainableModel):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(num_classes=2)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

        self.conv1 = nn.Conv2d(3, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 32, (3, 3))
        self.conv3 = nn.Conv2d(32, 2, (3, 3))

    def forward(self, x):
        # x = F.avg_pool2d(x, 2)
        # x = F.dropout(x, 0.3)
        # x = F.avg_pool2d(x, 2)
        # x = F.dropout(x, 0.3)
        # x = F.avg_pool2d(x, 2)
        # x = F.dropout(x, 0.3)
        # x = F.avg_pool2d(x, 2)
        # x = F.dropout(x, 0.3)
        # x = F.avg_pool2d(x, 2)
        # x = F.dropout(x, 0.3)
        # x = F.avg_pool2d(x, 2)
        # x = F.dropout(x, 0.3)
        # x = F.upsample(x, scale_factor=64, mode='bilinear')
        # x = self.resnet(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=2).mean(dim=2)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss, (loss.detach(),)