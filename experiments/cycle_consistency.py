
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet
from sklearn.model_selection import train_test_split
from fire import Fire

import IPython


def main(mse_weight=1, cycle_weight=1):

    # MODEL
    print ("Using UNet")
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
    def curvature_model(pred):
        return checkpoint(curvature_model_base, pred)

    inverse_model_base = DataParallelModel.load(UNet().cuda(), f"{MODELS_DIR}/results_inverse_cycle_unet1x1model.pth")
    def inverse_model(pred):
        return checkpoint(inverse_model_base, pred)

    def cycle_model(pred):
        return inverse_model(curvature_model(pred))

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        cycle = F.mse_loss(cycle_model(pred)*mask.float(), cycle_model(target)*mask.float())

        return mse_weight*mse + cycle_weight*cycle, (mse.detach(), cycle.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    def jointplot1(data):
        data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
        logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

    def jointplot2(data):
        data = np.stack((logger.data["train_cycle_loss"], logger.data["val_cycle_loss"]), axis=1)
        logger.plot(data, "cycle_loss", opts={"legend": ["train_cycle", "val_cycle"]})

    logger.add_hook(jointplot1, feature="val_mse_loss", freq=1)
    logger.add_hook(jointplot2, feature="val_cycle_loss", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502, 
        loss_models={"curvature": curvature_model, "cycle": cycle_model})

    # TRAINING
    for epochs in range(0, 800):
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (mse_data, cycle_data,) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_mse_loss", np.mean(mse_data))
        logger.update("train_cycle_loss", np.mean(cycle_data))

        val_set = itertools.islice(val_loader, val_step)
        (mse_data, cycle_data,) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_mse_loss", np.mean(mse_data))
        logger.update("val_cycle_loss", np.mean(cycle_data))

        plot_images(model, logger, test_set, ood_images, mask_val=0.502, 
                        loss_models={"curvature": curvature_model, "cycle": cycle_model})

        scheduler.step()


if __name__ == "__main__":
    Fire(main)
