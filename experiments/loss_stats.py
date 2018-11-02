
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
from modules.percep_nets import DenseNet, DeepNet, BaseNet
from modules.depth_nets import UNetDepth
from sklearn.model_selection import train_test_split
from fire import Fire

import IPython

def main():

    ##### DEFINE PERCEP LOSSES AND BASE LOSS HERE #####

    base_name = 'rgb_normal'
    base_model = DataParallelModel.load(ResNet().cuda(), f"/home/rohan/scaling/mount/shared/models/rgb2normal_256_32.pth")
    percep_losses = {'norm_curv': (f'/home/rohan/scaling/mount/shared/models/normal2curvature_dense_v2.pth', DenseNet), \
                     'depth_curv': (f'/home/rohan/scaling/mount/shared/models/normal2zdepth_unet.pth', UNetDepth)}

    ###################################################
    base_model.compile(torch.optim.Adam, lr=5e-4, weight_decay=2e-6, amsgrad=True)

    loss_stats = {name:[] for name in percep_losses.keys()}
    loss_stats[base_name] = []

    for name, (path, model) in list(percep_losses.items()):
        percep_losses[name] = DataParallelModel.load(model().cuda(), path)

    print (base_model.forward(torch.randn(8, 3, 256, 256)).shape)
    print (base_model.forward(torch.randn(16, 3, 256, 256)).shape)
    print (base_model.forward(torch.randn(32, 3, 256, 256)).shape)
    
    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        base_mse = F.mse_loss(pred*mask.float(), target*mask.float())
        mse_list = [(base_name, base_mse.detach())]
        for name, model in percep_losses.items():
            mse_percep = F.mse_loss(model(pred)*mask.float(), model(target)*mask.float())
            mse_list.append((name, mse_percep.detach()))
        return base_mse, mse_list

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    logger.add_hook(lambda x: pickle.dump(loss_stats, open(f"{RESULTS_DIR}/loss_stats.p", "wb")), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)

    # TRAINING
    for epochs in range(0, 800):

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        all_losses = base_model.predict_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )

        for x in all_losses:
            for name, mse in x:
                loss_stats[name].append(mse.data.cpu().numpy())
        for name in loss_stats:
            logger.text(f'{name} mean: {np.mean(loss_stats[name])}')

if __name__ == "__main__":
    Fire(main)
