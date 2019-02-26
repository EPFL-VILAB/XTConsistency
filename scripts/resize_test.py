
import os, sys, math, random, itertools
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from modules.percep_nets import DenseNet, DeepNet, BaseNet
from modules.depth_nets import UNetDepth
from sklearn.model_selection import train_test_split
from fire import Fire

import IPython

def main():

    model = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/augmented_base.pth")
    model.compile(torch.optim.Adam, lr=5e-4, weight_decay=2e-6, amsgrad=True)

    print (model.forward(torch.randn(8, 3, 256, 256)).shape)
    print (model.forward(torch.randn(16, 3, 256, 256)).shape)
    print (model.forward(torch.randn(32, 3, 256, 256)).shape)
    
    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        return mse, (mse.detach(),)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    for resize in range(128, 512+1, 64):
        # DATA LOADING
        train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
            load_data("rgb", "normal", batch_size=48, resize=resize)
        preds, targets, losses, _ = model.predict_with_data(test_set)
        logger.images(preds.clamp(min=0, max=1), f"predictions_{resize}", nrow=2, resize=resize)
        ood_preds = model.predict(ood_images)
        logger.images(ood_preds, f"ood_predictions_{resize}", nrow=2, resize=resize)

        if resize == 256:
            logger.images(targets.clamp(min=0, max=1), "targets", nrow=2, resize=256)

if __name__ == "__main__":
    Fire(main)
