
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
from .train_normal_curvature import Network as CurvatureNetwork

from modules.resnet import ResNet

from sklearn.model_selection import train_test_split
from fire import Fire

import IPython

def main(perceptual_weight=0, mse_weight=1, weight_step=None):

    # MODEL
    model = DataParallelModel(ResNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # PERCEPTUAL LOSS
    loss_model = DataParallelModel.load(CurvatureNetwork().cuda(), "/models/normal2curvature_v2.pth")

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        percep = F.mse_loss(loss_model(pred)*mask.float(), loss_model(target)*mask.float())
        return mse_weight*mse + perceptual_weight*percep, (mse.detach(), percep.detach())

    # LOGGING
    logger = VisdomLogger("train", server="35.230.67.129", port=7000, env=JOB)
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
    buildings = [file[6:-7] for file in glob.glob("/data/*_normal")]
    train_buildings, test_buildings = train_test_split(buildings, test_size=0.1)

    train_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=train_buildings, source_task="rgb", dest_task="normal"),
        batch_size=48,
        num_workers=16,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=test_buildings, source_task="rgb", dest_task="normal"),
        batch_size=48,
        num_workers=16,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=["almena"], source_task="rgb", dest_task="normal"),
        batch_size=12,
        num_workers=12,
        shuffle=False,
    )
    test_set = list(itertools.islice(test_loader, 1))
    test_images = torch.cat([x for x, y in test_set], dim=0)
    logger.images(test_images, "images", resize=128)

    logger.text("Train files count: " + str(len(train_loader.dataset)))
    logger.text("Val files count: " + str(len(val_loader.dataset)))

    train_loader, val_loader = cycle(train_loader), cycle(val_loader)

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
                mask = build_mask(target, val=0.502)
                mse = F.mse_loss(pred[mask], target[mask])
                percep = F.mse_loss(loss_model(pred)[mask], loss_model(target)[mask])
                return mse_weight*mse + perceptual_weight*percep, (mse.detach(), percep.detach())

        preds, targets, losses, _ = model.predict_with_data(test_set)
        test_masks = build_mask(targets, val=0.502)
        logger.images(test_masks.float(), "masks", resize=128)
        logger.images(preds, "predictions", nrow=1, resize=512)
        logger.images(targets, "targets", nrow=1, resize=512)

        with torch.no_grad():
            curvature_preds = loss_model(preds)
            curvature_targets = loss_model(targets)
            logger.images(curvature_preds, "curvature_predictions", resize=128)
            logger.images(curvature_targets, "curvature_targets", resize=128)

        scheduler.step()


if __name__ == "__main__":
    Fire(main)
