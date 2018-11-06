
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

from modules.unet import UNet

import IPython


if __name__ == "__main__":

    # MODEL
    print ("Using PyramidNet")
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    print (model.forward(torch.randn(1, 3, 512, 512)).shape)

    def loss(pred, target):
        mask = build_mask(target, val=0.0, tol=1e-2)
        mse = F.mse_loss(pred[mask], target[mask])
        unmask_mse = F.mse_loss(pred, target)
        return mse, (mse.detach(), unmask_mse.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature='loss', freq=25)

    def jointplot1(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    def jointplot2(data):
        data = np.stack([logger.data["train_unmask_loss"], logger.data["val_unmask_loss"]], axis=1)
        logger.plot(data, "unmask_loss", opts={'legend': ['train_unmask', 'val_unmask']})

    logger.add_hook(jointplot1, feature='val_loss', freq=1)
    logger.add_hook(jointplot2, feature='val_unmask_loss', freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature='loss', freq=400)

    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("normal", "reshading", batch_size=48)
    logger.images(test_images, "images", resize=128)
    plot_images(model, logger, test_set, mask_val=0.0)

    # TRAINING
    for epochs in range(0, 800):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, train_step)
        (losses, unmask_losses) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)
        logger.update('train_loss', np.mean(losses))
        logger.update('train_unmask_loss', np.mean(unmask_losses))

        val_set = itertools.islice(val_loader, val_step)
        (losses, unmask_losses) = model.predict_with_metrics(val_set, loss_fn=loss, logger=logger)
        logger.update('val_loss', np.mean(losses))
        logger.update('val_unmask_loss', np.mean(unmask_losses))

        plot_images(model, logger, test_set, mask_val=0.0)
