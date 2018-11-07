
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
from modules.unet import UNet
from skimage import feature
from scipy import ndimage
import IPython


if __name__ == "__main__":

    # MODEL
    model = DataParallelModel(UNet(downsample=4, out_channels=1))
    model.compile(torch.optim.Adam, lr=2e-4, weight_decay=2e-6, amsgrad=True)
    print (model.forward(torch.randn(1, 3, 256, 256)).shape)
    
    filter = gaussian_filter(kernel_size=5, sigma=2).float()
    def dest_transforms(x):


        image = x.data.cpu().numpy().mean(axis=0)
        blur = ndimage.filters.gaussian_filter(image, sigma=2, )
        sx = ndimage.sobel(blur, axis=0, mode='constant')
        sy = ndimage.sobel(blur, axis=1, mode='constant')
        sob = np.hypot(sx, sy)
        # edge = feature.canny(image, sigma=3.0)*1.0
        # edge = feature.canny(image, sigma=3.0, low_threshold=None, high_threshold=None, use_quantiles=True)*1.0
        edge = torch.FloatTensor(sob).unsqueeze(0)
        # with torch.no_grad():
        #     edge = F.conv2d(edge.unsqueeze(0), weight=filter, bias=None, groups=1, padding=2, stride=1)[0]
        return edge

    print (dest_transforms(torch.randn(3, 256, 256)).shape)

    def loss(pred, target):
        mse = F.mse_loss(pred, target)
        return mse, (mse.detach(),)

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
        load_data("normal", "rgb", batch_size=64, dest_transforms=dest_transforms)
    logger.images(test_images, "images", resize=128)
    plot_images(model, logger, test_set, mask_val=-1.0)
    print("starting training...")
    # TRAINING
    for epochs in range(0, 200):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, train_step)
        (losses,) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)
        logger.update('train_loss', np.mean(losses))

        val_set = itertools.islice(val_loader, val_step)
        (losses,) = model.predict_with_metrics(val_set, loss_fn=loss, logger=logger)
        logger.update('val_loss', np.mean(losses))

        plot_images(model, logger, test_set, mask_val=-1)
