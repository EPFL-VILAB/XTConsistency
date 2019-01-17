
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

from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset
import task_configs

from fire import Fire
import IPython
import task_configs
from task_configs import Task
from sklearn.metrics import jaccard_similarity_score as jsc
from sklearn.metrics import accuracy_score as acc_score

def main():

    def loss(pred, target):
        mse = ((pred-target)**2).mean()
        return mse, (mse.detach(), 0, 0)
    def val_loss(pred, target):
        mse = ((pred-target)**2).mean()
        pred_bin = (pred > 0.5).data.cpu().numpy().reshape(-1)
        target_bin = (target > 0.5).data.cpu().numpy().reshape(-1)
        iou = jsc(pred_bin, target_bin)
        acc = acc_score(pred_bin, target_bin)
        return mse, (mse.detach(), iou, acc)
    # MODEL
    src_task = Task('original', shape=(3, 256, 256))
    dest_task = Task('mask', shape=(1, 256, 256))

    model = DataParallelModel((task_configs.get_model(src_task, dest_task)).cuda())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="mse_loss"), feature="val_mse_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="iou_loss"), feature="val_iou_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="acc_loss"), feature="val_acc_loss", freq=1)

    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/{src_task.name}2{dest_task.name}.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data(src_task, dest_task, batch_size=48)
    logger.images(test_images, "images", resize=128)

    for epochs in range(0, 800):

        plot_images(model, logger, test_set, mask_val=dest_task.mask_val)

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        val_set = itertools.islice(val_loader, val_step)

        (train_mse_data, train_iou, train_acc) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)
        logger.update("train_mse_loss", np.mean(train_mse_data))
        logger.update("train_iou_loss", np.mean(train_iou))
        logger.update("train_acc_loss", np.mean(train_acc))

        (val_mse_data, val_iou, val_acc) = model.predict_with_metrics(val_set, loss_fn=val_loss, logger=logger)
        logger.update("val_mse_loss", np.mean(val_mse_data))
        logger.update("val_iou_loss", np.mean(val_iou))
        logger.update("val_acc_loss", np.mean(val_acc))

        


if __name__ == "__main__":
    Fire(main)
