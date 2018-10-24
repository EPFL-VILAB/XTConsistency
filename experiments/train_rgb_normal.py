
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
from modules.percep_nets import DenseNet, DeepNet, BaseNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet
from sklearn.model_selection import train_test_split
from fire import Fire

import IPython


def main(curvature_step=0, depth_step=0):

    curvature_weight = 0.0
    depth_weight = 0.0

    # MODEL
    model = DataParallelModel(ResNet())
    # model = DataParallelModel.load(ResNet().cuda(), f"{RESULTS_DIR}/model.pth")
    model.compile(torch.optim.Adam, lr=5e-4, weight_decay=2e-6, amsgrad=True)

    print (model.forward(torch.randn(8, 3, 256, 256)).shape)
    print (model.forward(torch.randn(16, 3, 256, 256)).shape)
    print (model.forward(torch.randn(32, 3, 256, 256)).shape)
    # print (model.forward(torch.randn(64, 3, 256, 256)).shape)
    # print (model.forward(torch.randn(128, 3, 256, 256)).shape)
    # print (model.forward(torch.randn(256, 3, 256, 256)).shape)
    
    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    def jointplot1(data):
        data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
        logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

    logger.add_hook(jointplot1, feature="val_mse_loss", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=64)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=64)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502)

    # TRAINING
    for epochs in range(0, 800):

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (train_mse_data,) = model.fit_with_metrics(train_set, logger=logger)

        logger.update("train_mse_loss", np.mean(train_mse_data))

        val_set = itertools.islice(val_loader, val_step)
        (val_mse_data,) = model.predict_with_metrics(val_set, logger=logger)
        logger.update("val_mse_loss", np.mean(val_mse_data))

        #print out running variances
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                print(f"running_var: {str(m.running_var.mean())}")
                logger.text(f"running_var: {str(m.running_var.mean().data.cpu().numpy())}")

        # stop if we get a high val mse
        if np.mean(val_mse_data) - np.mean(train_mse_data) > 0.01:
            print("high val mse!!!")
            logger.text("high val mse!!!")
            IPython.embed()

        plot_images(model, logger, test_set, ood_images, mask_val=0.502)


if __name__ == "__main__":
    Fire(main)
