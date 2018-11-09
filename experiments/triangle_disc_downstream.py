
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


def main(disc_step=0.01):

    curvature_weight = 0.0
    depth_weight = 1
    disc_weight = 0.0

    # MODEL
    print ("Using UNet")
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # use discriminator
    disc = DataParallelModel(ResNetDisc())
    disc.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    # print(disc.forward(torch.randn(8, 3, 256, 256)))

    def mixed_loss(pred, target, data):
        mask = build_mask(target.detach(), val=0.502)

        dcycle_pred = depth_cycle(pred).expand_as(target)
        dmodel_pred = depth_model(pred).expand_as(target)
        dmodel_target = depth_model(target).expand_as(target)

        labels = torch.tensor(0, device=pred.device).expand(pred.shape[0]*3)
        labels[:2*pred.shape[0]] = 1
        _, disc_loss, _ = disc.fit_on_batch(torch.cat([dcycle_pred.detach(), dmodel_pred.detach(), dmodel_target.detach()]), labels, train=(pred.requires_grad))

        mse_loss = lambda x, y: ((x-y)**2).mean()
        # mse = mse_loss(pred*mask.float(), target*mask.float())
        # inverse = mse_loss(normal2rgb(pred)*mask.float(), data.to(pred.device)*mask.float())
        
        disc_loss1, _ = disc.loss(disc.forward(dcycle_pred), torch.tensor(0, device=pred.device).expand(pred.shape[0]))
        disc_loss2, _ = disc.loss(disc.forward(dmodel_pred), torch.tensor(0, device=pred.device).expand(pred.shape[0]))
        disc_loss = disc_loss1 + disc_loss2

        depth = torch.tensor(0.0, device=pred.device) if depth_weight == 0.0 else \
                    mse_loss(dcycle_pred*mask.float(), dmodel_pred*mask.float())

        return disc_weight*disc_loss + depth_weight*depth, (disc_loss.detach(), depth.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=10)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="disc_loss"), feature="val_disc_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="depth_loss"), feature="val_depth_loss", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=32)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502, 
        loss_models={"curvature": curvature_model, "depth": depth_model, "depth_cycle": depth_cycle})

    # TRAINING
    for epochs in range(0, 800):
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (disc_data, depth_data) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_disc_loss", np.mean(disc_data))
        logger.update("train_depth_loss", np.mean(depth_data))

        val_set = itertools.islice(val_loader, val_step)
        (disc_data, depth_data) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_disc_loss", np.mean(disc_data))
        logger.update("val_depth_loss", np.mean(depth_data))

        disc_weight += disc_step
        logger.text (f"Increasing discriminator weight: {disc_weight}")

        plot_images(model, logger, test_set, ood_images, mask_val=0.502, 
            loss_models={"curvature": curvature_model, "depth": depth_model, "depth_cycle": depth_cycle})

        scheduler.step()


if __name__ == "__main__":
    Fire(main)