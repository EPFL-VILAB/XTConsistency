
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

    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x)

    def loss(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss, (loss.detach(),)