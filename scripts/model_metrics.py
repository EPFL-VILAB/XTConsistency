
import os, sys, math, random, itertools
import numpy as np
import pickle
from time import sleep
from collections import defaultdict


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset, ImageDataset, GeneralTaskLoader

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from modules.percep_nets import DenseNet, DeepNet, BaseNet
from modules.depth_nets import UNetDepth
from sklearn.model_selection import train_test_split
from fire import Fire

from transfers import finetuned_transfers
from task_configs import TASK_MAP
from transfers import functional_transfers

import IPython

TRANSFER_MAP = {(t.src_task.name, t.dest_task.name):t for t in functional_transfers}


def main():


    models = [
        ('unet_baseline', lambda: UNetOld(), f"{MODELS_DIR}/unet_baseline.pth"),
        ('unet_percepstep_0.1', lambda: UNetOld(), f"{MODELS_DIR}/unet_percepstep_0.1.pth"),
        ('mixing_percepcurv_norm', lambda: UNet(), f"{MODELS_DIR}/mixing_percepcurv_norm.pth"),
        ('unet_percep_epoch150_100', lambda: UNetOld(), f"{MODELS_DIR}/unet_percep_epoch150_100.pth"),
        # # ('rgb2normal_multitask', lambda: UNet(in_channels=3, out_channels=6), f"{MODELS_DIR}/rgb2normal_multitask.pth"),
        ('rgb2normal_random', lambda: UNet(), f"{MODELS_DIR}/rgb2normal_random.pth"),
    ]

    # model = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/mixing_percepcurv_norm.pth")
    # model.compile(torch.optim.Adam, lr=5e-4, weight_decay=2e-6, amsgrad=True)

    # print (model.forward(torch.randn(8, 3, 256, 256)).shape)
    # print (model.forward(torch.randn(16, 3, 256, 256)).shape)
    # print (model.forward(torch.randn(32, 3, 256, 256)).shape)
    
    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)
    resize = 256
    tasks = [TASK_MAP[name] for name in ['rgb', 'normal', 'principal_curvature', 'depth_zbuffer']]

    test_loader = torch.utils.data.DataLoader(
        GeneralTaskLoader(['almena'], tasks),
        batch_size=64,
        num_workers=12,
        shuffle=False,
        pin_memory=True
    )
    imgs = list(itertools.islice(test_loader, 1))[0]
    gt = {tasks[i].name:batch.cuda() for i, batch in enumerate(imgs)}

    def mse_func(x, y):
        return ((x - y)**2).mean().data.cpu().numpy()

    with torch.no_grad():
        for name, model, path in models:
            print('testing, ', name)
            model = DataParallelModel.load(model().cuda(), path)
            normal_preds = model.forward(gt['rgb'])
            normal_mse = mse_func(gt['normal'], normal_preds)
            print(f'normal mse: {normal_mse}')
            for task in tasks:
                if task.name == 'rgb' or task.name == 'normal': continue
                transfer_model = TRANSFER_MAP[('normal', task.name)]
                task_preds = transfer_model(normal_preds)
                percep_mse = mse_func(task_preds, transfer_model(gt['normal']))
                print(f'{task.name} percep_loss: {percep_mse}')
                gt_mse = mse_func(task_preds, gt[task.name])
    

if __name__ == "__main__":
    Fire(main)
