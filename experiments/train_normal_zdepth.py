
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
            
    def __init__(self, f1, f2, dilation=1, transpose=False):
        super().__init__()
        self.transpose = transpose
        self.conv = nn.Conv2d(f1, f2, (3, 3), padding=1)
        if self.transpose:
            self.convt = nn.ConvTranspose2d(f1, f1, (3, 3), dilation=dilation,
                stride=2, padding=dilation, output_padding=1)
        self.bn = nn.BatchNorm2d(f1)
        
    def forward(self, x):
        #x = F.dropout(x, 0.04, self.training)
        x = self.bn(x)
        if self.transpose:
            x = F.relu(self.convt(x))
        x = F.relu(self.conv(x))
        return x


class Network(TrainableModel):

    def __init__(self):
        super(Network, self).__init__()
        self.decoder = nn.Sequential(ConvBlock(3, 32),
                            ConvBlock(32, 32), ConvBlock(32, 32, dilation=2),
                            ConvBlock(32, 1, dilation=4)
                        )

    def forward(self, x):
        x = self.decoder(x)
        return x

    def loss(self, pred, target):
        mask = build_mask(target)
        # print ("Mask: ", mask.float().sum()/mask.nelement())
        return 256.0*F.mse_loss(pred[mask], target[mask])



if __name__ == "__main__":

    # MODEL
    model = DataParallelModel(Network())
    model.compile(torch.optim.Adam, lr=2e-4, weight_decay=2e-6, amsgrad=True)
    # scheduler = MultiStepLR(model.optimizer, milestones=[5*i+1 for i in range(0, 80)], gamma=0.85)
    # print (model.forward(torch.randn(1, 3, 512, 512)).shape)

    # LOGGING
    logger = VisdomLogger("train", server='35.230.67.129', port=7000, env=JOB)
    logger.add_hook(lambda x: logger.step(), feature='loss', freq=25)

    def jointplot(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    logger.add_hook(jointplot, feature='val_loss', freq=1)
    logger.add_hook(lambda x: 
        [print ("Saving model to /result/model.pth"),
        model.save("/result/model.pth")],
        feature='loss', freq=400,
    )

     # DATA LOADING
    buildings = [file[6:-7] for file in glob.glob("/data/*_normal")]
    train_buildings, test_buildings = train_test_split(buildings, test_size=0.1)

    to_tensor = transforms.ToTensor()
    dest_transforms = lambda x: to_tensor(x).float()/256**2

    train_loader = torch.utils.data.DataLoader(
                            ImageTaskDataset(buildings=train_buildings, 
                                source_task='normal', dest_task='depth_zbuffer',
                                dest_transforms=dest_transforms),
                        batch_size=80, num_workers=16, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
                            ImageTaskDataset(buildings=test_buildings, 
                                source_task='normal', dest_task='depth_zbuffer',
                                dest_transforms=dest_transforms),
                        batch_size=80, num_workers=16, shuffle=True)

    logger.text("Train files count: " + str(len(train_loader.dataset)))
    logger.text("Val files count: " + str(len(val_loader.dataset)))

    train_loader, val_loader = cycle(train_loader), cycle(val_loader)

    # TRAINING
    for epochs in range(0, 100):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, 200)
        (losses,) = model.fit_with_metrics(train_set, logger=logger, metrics=[model.loss])
        logger.update('train_loss', np.mean(losses))

        val_set = itertools.islice(val_loader, 200)
        (losses,) = model.predict_with_metrics(val_set, logger=logger, metrics=[model.loss])
        logger.update('val_loss', np.mean(losses))

        test_set = list(itertools.islice(val_loader, 1))
        test_images = torch.cat([x for x, y in test_set], dim=0)
        preds, targets, losses, _ = model.predict_with_data(test_set)
        print (targets.shape)
        test_masks = build_mask(targets)

        logger.images(test_images, "images")
        logger.images(test_masks, "masks")
        logger.images(preds, "predictions")
        logger.images(targets, "targets")
