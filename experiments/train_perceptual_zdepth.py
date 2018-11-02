
import os, sys, math, random, itertools
import numpy as np

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
from modules.depth_nets import UNetDepth
from modules.unet import UNet
from sklearn.model_selection import train_test_split
from fire import Fire

import IPython


def main(perceptual_weight=0, mse_weight=1, weight_step=None):

    # MODEL
    model = DataParallelModel(ResNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    
    print (model.forward(torch.randn(1, 3, 512, 512)).shape)
    print (model.forward(torch.randn(8, 3, 512, 512)).shape)
    print (model.forward(torch.randn(16, 3, 512, 512)).shape)
    print (model.forward(torch.randn(24, 3, 512, 512)).shape)
    print (model.forward(torch.randn(32, 3, 512, 512)).shape)
    
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # loss_model = DataParallelModel.load(DenseNet().cuda(), "/models/normal2curvature_dense.pth")
    # loss_model = DataParallelModel.load(DeepNet().cuda(), "/models/normal2curvature_deep.pth")
    loss_model = DataParallelModel.load(UNetDepth().cuda(), "/models/normal2zdepth_unet.pth")

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        percep = torch.tensor(0.0).to(mse.device) if perceptual_weight == 0 else F.mse_loss(loss_model(pred)*mask.float(), loss_model(target)*mask.float())
        return mse_weight*mse + perceptual_weight*percep, (mse.detach(), percep.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    def jointplot1(data):
        data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
        logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

    def jointplot2(data):
        data = np.stack((logger.data["train_perceptual_loss"], logger.data["val_perceptual_loss"]), axis=1)
        logger.plot(data, "perceptual_loss", opts={"legend": ["train_perceptual", "val_perceptual"]})

    logger.add_hook(jointplot1, feature="val_mse_loss", freq=1)
    logger.add_hook(jointplot2, feature="val_perceptual_loss", freq=1)
    logger.add_hook(lambda x: model.save("/result/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images = load_data("rgb", "normal", batch_size=16)
    logger.images(test_images, "images", resize=128)
    print (len(ood_images))
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502, loss_model=loss_model)

    # TRAINING
    for epochs in range(0, 800):

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, 200)
        (mse_data, perceptual_data) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_mse_loss", np.mean(mse_data))
        logger.update("train_perceptual_loss", np.mean(perceptual_data))

        val_set = itertools.islice(val_loader, 200)
        (mse_data, perceptual_data) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_mse_loss", np.mean(mse_data))
        logger.update("val_perceptual_loss", np.mean(perceptual_data))

        if weight_step is not None:
            perceptual_weight += weight_step
            logger.text (f"Increasing perceptual loss weight: {perceptual_weight}")
            
            def mixed_loss(pred, target):
                mask = build_mask(target.detach(), val=0.502)
                mse = F.mse_loss(pred*mask.float(), target*mask.float())
                percep = torch.tensor(0.0).to(mse.device) if perceptual_weight == 0 else F.mse_loss(loss_model(pred)*mask.float(), loss_model(target)*mask.float())
                return mse_weight*mse + perceptual_weight*percep, (mse.detach(), percep.detach())

        plot_images(model, logger, test_set, ood_images, mask_val=0.502, loss_model=loss_model)

        scheduler.step()


if __name__ == "__main__":
    Fire(main)
