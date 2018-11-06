
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

from modules.percep_nets import DenseNet, DeepNet, BaseNet, ResidualsNet, WideNet, PyramidNet

import IPython

def gaussian_filter(kernel_size=5, sigma=1.0, device=0):

    channels = 1
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size).float().to(device)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel.to(device) / torch.sum(gaussian_kernel).to(device)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel

if __name__ == "__main__":

    # MODEL
    model = DataParallelModel(BaseNet())
    model.compile(torch.optim.Adam, lr=2e-4, weight_decay=2e-6, amsgrad=True)
    print (model.forward(torch.randn(1, 3, 512, 512)).shape)

    def dest_transforms(x):
        x = x.unsqueeze(0)
        filter = gaussian_filter(kernel_size=5, sigma=2, device=x.device).float()
        with torch.no_grad():
            x = F.conv2d(x.float(), weight=filter.to(x.device), bias=None, groups=1, padding=2, stride=1)
        return x[0]

    def loss(pred, target):
        mse = F.mse_loss(pred, target)
        unmask_mse = F.mse_loss(pred, target)
        return mse, (mse.detach(), unmask_mse.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature='loss', freq=25)

    def jointplot1(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    logger.add_hook(jointplot1, feature='val_loss', freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/basenet.pth"), feature='loss', freq=400)

    print("about to load data...")
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("normal", "edge_texture", batch_size=64)
    logger.images(test_images, "images", resize=128)
    plot_images(model, logger, test_set, mask_val=0.0)
    print("starting training...")
    # TRAINING
    for epochs in range(0, 200):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, train_step)
        (losses, unmask_losses) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)
        logger.update('train_loss', np.mean(losses))

        val_set = itertools.islice(val_loader, val_step)
        (losses, unmask_losses) = model.predict_with_metrics(val_set, loss_fn=loss, logger=logger)
        logger.update('val_loss', np.mean(losses))

        plot_images(model, logger, test_set, mask_val=-1)
