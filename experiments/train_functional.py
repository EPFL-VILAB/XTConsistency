
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from plotting import *
from functional import get_functional_loss
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from fire import Fire

from skimage import feature
from functools import partial
from scipy import ndimage

import IPython


def main(loss_config="gt_mse", mode="standard"):

    # FUNCTIONAL LOSS
    functional = get_functional_loss(config=loss_config, mode=mode)
    print ("Losses: ", functional.losses.keys())

    # MODEL
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=10)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)
    logger.add_hook(lambda x: scheduler.step(), feature="epoch", freq=1)
    functional.logger_hooks(logger)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=64)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    # TRAINING
    for epochs in range(0, 800):

        plot_images(model, logger, test_set, ood_images, mask_val=functional.dest_task.mask_val, 
            loss_models=functional.plot_losses,
        )
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, 10)
        val_set = itertools.islice(val_loader, 2)

        train_metrics = model.fit_with_metrics(train_set, loss_fn=functional, logger=logger)
        val_metrics = model.predict_with_metrics(val_set, loss_fn=functional, logger=logger)
        functional.logger_update(logger, train_metrics, val_metrics)


if __name__ == "__main__":
    Fire(main)
