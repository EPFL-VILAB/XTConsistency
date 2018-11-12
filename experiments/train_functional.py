
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
from transfers import functional_transfers
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


def main(curvature_step=0, depth_step=0):

    curvature_weight = curvature_step
    depth_weight = depth_step

    # MODEL
    print ("Using UNet")
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    loss_names = ["gt_percep",]

    def mixed_loss(y, y_hat, x):
        mask = build_mask(y_hat.detach(), val=0.502)
        mse_loss = lambda x, y: ((x*mask.float() - y*mask.float())**2).mean()
        (f, F, g, G, s, CE, EC, DE, a) = functional_transfers
        print ("Sobel: ", a(x).shape)
        losses = [
            mse_loss(f(y), f(y_hat)),
        ]
        return sum(losses), (loss.detach() for loss in losses)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=10)
    for loss in loss_names:
        logger.add_hook(partial(jointplot, logger=logger, loss_type=f"{loss}"), feature=f"val_{loss}", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    # TRAINING
    for epochs in range(0, 800):

        (f, F, g, G, s, CE, EC, DE, a) = functional_transfers
        plot_images(model, logger, test_set, ood_images, mask_val=-1.0, 
            loss_models={
                "f(y)": lambda y, y_hat, x: f(y), 
                "f(y_hat)": lambda y, y_hat, x: f(y_hat), 
                "F(f(y))": lambda y, y_hat, x: F(f(y)), 
            },
        )
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        loss_metrics = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        for loss, metric in zip(loss_names, loss_metrics):
            logger.update(f"train_{loss}", np.mean(metric))

        val_set = itertools.islice(val_loader, val_step)
        loss_metrics = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        for loss, metric in zip(loss_names, loss_metrics):
            logger.update(f"val_{loss}", np.mean(metric))
        scheduler.step()


if __name__ == "__main__":
    Fire(main)
