
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
from modules.percep_nets import DenseNet, DeepNet, BaseNet, Dense1by1Net
from modules.depth_nets import UNetDepth
from modules.unet import UNet
from sklearn.model_selection import train_test_split
from fire import Fire

from torch.utils.checkpoint import checkpoint
import task_configs
import IPython


def main(src_task, dest_task):

    # MODEL
    task_map = task_configs.create_tasks()

    src_task = task_map[src_task]
    dest_task = task_map[dest_task]

    model = DataParallelModel((task_configs.get_model(src_task, dest_task)).cuda())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    def loss(pred, target):
        # print(target.shape)
        # print(pred.shape)
        curv_target, normal_target = target[:,:3,:,:], target[:,3:,:,:]
        curv_pred, normal_pred = pred[:,:3,:,:], pred[:,3:,:,:]
        
        curv_mask = build_mask(curv_target.detach(), val=0.0)
        normal_mask = build_mask(normal_target.detach(), val=0.502)

        curv_mse = F.mse_loss(curv_pred*curv_mask.float(), curv_target*curv_mask.float())
        normal_mse = F.mse_loss(normal_pred*normal_mask.float(), normal_target*normal_mask.float())

        return curv_mse + normal_mse, (curv_mse.detach(), normal_mse.detach())

    # print (model(torch.randn(8, 3, 256, 256)).shape)
    # print (model(torch.randn(16, 3, 256, 256)).shape)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    def jointplot1(data):
        data = np.stack((logger.data["train_curvmse_loss"], logger.data["val_curvmse_loss"]), axis=1)
        logger.plot(data, "train_mse_loss", opts={"legend": ["train_curvmse_loss", "val_curvmse_loss"]})
    def jointplot2(data):
        data = np.stack((logger.data["train_normmse_loss"], logger.data["val_normmse_loss"]), axis=1)
        logger.plot(data, "curve_mse_loss", opts={"legend": ["train_normmse_loss", "val_normmse_loss"]})

    logger.add_hook(jointplot1, feature="val_curvmse_loss", freq=1)
    logger.add_hook(jointplot2, feature="val_normmse_loss", freq=1)

    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/{src_task.name}2{dest_task.name}.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data(src_task, dest_task, batch_size=48)
    logger.images(test_images, "images", resize=128)

    plot_images(model, logger, test_set, mask_val=dest_task.mask_val, target_plot_func=dest_task.plot_func)
    # TRAINING
    for epochs in range(0, 800):

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (train_curvmse_data, train_normmse_data) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)

        logger.update("train_curvmse_data", np.mean(train_curvmse_data))
        logger.update("train_normmse_data", np.mean(train_normmse_data))

        val_set = itertools.islice(val_loader, val_step)
        (val_curvmse_data, val_normmse_data) = model.predict_with_metrics(val_set, loss_fn=loss, logger=logger)
        logger.update("val_curvmse_data", np.mean(val_curvmse_data))
        logger.update("val_normmse_data", np.mean(val_normmse_data))

        plot_images(model, logger, test_set, mask_val=dest_task.mask_val, target_plot_func=dest_task.plot_func)


if __name__ == "__main__":
    Fire(main)
