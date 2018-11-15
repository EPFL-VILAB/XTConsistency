
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


def main(src_task, dest_task):

    # MODEL
    task_map = task_configs.create_tasks()

    src_task = task_map[src_task]
    dest_task = task_map[dest_task]

    model = DataParallelModel((task_configs.get_model(src_task, dest_task)).cuda())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="mse_loss"), feature="val_mse_loss", freq=1)
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

        (train_mse_data,) = model.fit_with_metrics(train_set, loss_fn=dest_task.loss_func, logger=logger)
        logger.update("train_mse_loss", np.mean(train_mse_data))
        
        (val_mse_data,) = model.predict_with_metrics(val_set, loss_fn=dest_task.loss_func, logger=logger)
        logger.update("val_mse_loss", np.mean(val_mse_data))

        


if __name__ == "__main__":
    Fire(main)
