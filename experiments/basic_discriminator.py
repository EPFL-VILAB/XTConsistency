
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
from transfers import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from modules.discriminator import ResNetDisc
from fire import Fire

from skimage import feature
from functools import partial

import IPython


def main(disc_step=0.0):

    disc_weight = 0.0

    # MODEL
    print ("Using UNet")
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # use discriminator
    disc = DataParallelModel(ResNetDisc())
    disc.compile(torch.optim.Adam, lr=1e-5, weight_decay=2e-6, amsgrad=True)
    # print(disc.forward(torch.randn(8, 3, 256, 256)))

    def mixed_loss(pred, target, data):
        mask = build_mask(target.detach(), val=0.502)

        labels = torch.tensor(0, device=pred.device).expand(pred.shape[0]*2)
        labels[:pred.shape[0]] = 1
        predhat, disc_loss, _ = disc.fit_on_batch(torch.cat([pred.detach(), target]), labels, train=(pred.requires_grad))
        accuracy = (torch.argmax(predhat, dim=1) == labels).sum()/(1.0*labels.shape[0])

        mse_loss = lambda x, y: ((x-y)**2).mean()
        mse = mse_loss(pred*mask.float(), target*mask.float())
        # inverse = mse_loss(normal2rgb(pred)*mask.float(), data.to(pred.device)*mask.float())
        
        predhat = disc.forward(pred)
        disc_loss, _ = disc.loss(predhat, torch.tensor(0, device=pred.device).expand(pred.shape[0]))
        # curvature = torch.tensor(0.0, device=pred.device) if curvature_weight == 0.0 else \
        #             mse_loss(curve_cycle(pred)*mask.float(), curvature_model(pred)*mask.float())
        # depth = torch.tensor(0.0, device=pred.device) if depth_weight == 0.0 else \
        #             mse_loss(depth_cycle(pred)*mask.float(), depth_model(pred)*mask.float())
        return mse + disc_weight*disc_loss, (mse.detach(), disc_loss.detach(), accuracy.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=10)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="mse_loss"), feature="val_mse_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="disc_loss"), feature="val_disc_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="accuracy"), feature="val_accuracy", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502)

    # TRAINING
    for epochs in range(0, 800):
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (mse_data, disc_data, accuracy_data) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_mse_loss", np.mean(mse_data))
        logger.update("train_disc_loss", np.mean(disc_data))
        logger.update("train_accuracy", np.mean(accuracy_data))

        val_set = itertools.islice(val_loader, val_step)
        (mse_data, disc_data, accuracy_data) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_mse_loss", np.mean(mse_data))
        logger.update("val_disc_loss", np.mean(disc_data))
        logger.update("val_accuracy", np.mean(accuracy_data))

        if epochs > 20:
            disc_step = 0.001
            logger.text ("Starting discriminator now")
        disc_weight += disc_step
        logger.text (f"Increasing discriminator weight: {disc_weight}")

        plot_images(model, logger, test_set, ood_images, mask_val=0.502)
        scheduler.step()


if __name__ == "__main__":
    Fire(main)