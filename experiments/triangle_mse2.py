
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

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        F, f, s, H_g = curvature2normal, curvature_model, normal2edge, curve_cycle
        ax, y, y_hat = target, pred, target
        mse_loss = lambda x, y: ((x*mask.float() -y*mask.float())**2).mean()


        cycle = mse_loss(F(H_g(y)), y)
        gt_cycle = mse_loss(F(H_g(y)), F(H_g(y_hat)))
        mse = mse_loss(y, y_hat)
        
        # curvature = torch.tensor(0.0, device=mse.device) if curvature_weight == 0.0 else \
        #     F.mse_loss(curvature_model(pred) * mask.float(), (target) * mask.float())
        # depth = torch.tensor(0.0, device=mse.device) if depth_weight == 0.0 else \
        #     F.mse_loss(depth_model(pred) * mask.float(), depth_model(target) * mask.float())

        return mse + cycle + gt_cycle, (mse.detach(), cycle.detach(), gt_cycle.detach())

    ### LOGGING ###
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=10)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    logger.add_hook(partial(jointplot, logger=logger, loss_type="mse_loss"), feature="val_mse_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="cycle_loss"), feature="val_cycle_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="gt_cycle_loss"), feature="val_gt_cycle_loss", freq=1)

    ### DATA LOADING ###
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=-1.0, 
            loss_models={"depth": depth_model, 
                        "depth2curve": curve_cycle,
                        "cycle": lambda x: curvature2normal(curve_cycle(x))}, loss_targets=False)

    for epochs in range(0, 800):
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (cycle_data, gt_cycle_data) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_cycle_loss", np.mean(cycle_data))
        logger.update("train_gt_cycle_loss", np.mean(gt_cycle_data))

        val_set = itertools.islice(val_loader, val_step)
        (cycle_data, gt_cycle_data) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_cycle_loss", np.mean(cycle_data))
        logger.update("val_gt_cycle_loss", np.mean(gt_cycle_data))

        plot_images(model, logger, test_set, ood_images, mask_val=-1.0, 
            loss_models={"depth": depth_model, 
                        "depth2curve": curve_cycle,
                        "cycle": lambda x: curvature2normal(curve_cycle(x))}, loss_targets=False)

        scheduler.step()
        logger.step()


if __name__ == "__main__":
    Fire(main)
