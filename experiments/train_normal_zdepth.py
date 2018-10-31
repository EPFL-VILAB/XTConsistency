
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

from modules.depth_nets import UNetDepth, ResNetDepth, ResidualsNet
from sklearn.model_selection import train_test_split

import IPython



if __name__ == "__main__":

    # MODEL
    model = DataParallelModel(UNetDepth())
    print ("Loader model")
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    print (model.forward(torch.randn(1, 3, 256, 256)).shape)

    def loss(pred, target):
        mask = build_mask(target, val=1.0, tol=1e-3)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        return mse, (mse.detach(),)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature='loss', freq=25)

    def jointplot(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    logger.add_hook(jointplot, feature='val_loss', freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature='loss', freq=400)

    # DATA LOADING
    def dest_transforms(x):
        return (x.float()/10000.0).clamp(min=0.0, max=1.0)

    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("normal", "depth_zbuffer", batch_size=64, dilate=5, dest_transforms=dest_transforms)
    logger.images(test_images, "images", resize=128)    
    plot_images(model, logger, test_set, mask_val=1.0)

    # TRAINING
    for epochs in range(0, 800):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, 200)
        (losses,) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)
        logger.update('train_loss', np.mean(losses))

        val_set = itertools.islice(val_loader, 200)
        (losses,) = model.predict_with_metrics(val_set, loss_fn=loss, logger=logger)
        logger.update('val_loss', np.mean(losses))

        plot_images(model, logger, test_set, mask_val=1.0)
