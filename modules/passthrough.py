
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import TrainableModel
from utils import *


class PassThroughModel(TrainableModel):
    def __init__(self, channel_range=(0, 3)):
        super().__init__()

        self.channel_range = channel_range
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.alpha * x[:, self.channel_range[0]:self.channel_range[1]] + self.beta

    def loss(self, pred, target):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)


