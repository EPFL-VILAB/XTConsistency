
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset
from torch.optim.lr_scheduler import MultiStepLR

from sklearn.model_selection import train_test_split

import IPython


class ConvBlock(nn.Module):
    def __init__(self, f1, f2, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (3, 3), padding=1)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(f1, f1, (3, 3), stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(f1)

    def forward(self, x):
        x = self.bn(x)
        if self.transpose:
            x = F.relu(self.convt(x))
        x = F.relu(self.conv(x))
        return x


class Network(TrainableModel):
    def __init__(self):
        super(Network, self).__init__()
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

    def loss(self, pred, target):
        mask = build_mask(pred, val=0.502)
        return F.mse_loss(pred[mask], target[mask])


if __name__ == "__main__":

    # MODEL
    model = DataParallelModel(Network())
    model.compile(torch.optim.Adam, lr=1e-4, weight_decay=2e-6, amsgrad=True)

    # LOGGING
    logger = VisdomLogger("train", server="35.230.67.129", port=7000, env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    # DATA LOADING
    buildings = [file[6:-7] for file in glob.glob("/data/*_normal")]
    train_buildings, test_buildings = train_test_split(buildings, test_size=0.1)

    train_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=["maven"], source_task="rgb", dest_task="normal"),
        batch_size=80,
        num_workers=16,
        shuffle=True,
    )
    train_loader = cycle(train_loader)

    # TRAINING
    for epochs in range(0, 800):

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, 200)
        (losses,) = model.fit_with_metrics(train_set, logger=logger, metrics=[model.loss])
        logger.update("train_loss", np.mean(losses))