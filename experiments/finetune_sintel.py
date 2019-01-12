
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *

from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from task_configs import get_task, get_model, tasks
from transfers import functional_transfers
from datasets import load_train_val, load_test, load_ood, load_sintel_train_val_test

from fire import Fire
import IPython


def main():

    src_task, dest_task = tasks.rgb, tasks.normal
    model = functional_transfers.n.load_model()
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=25)
    logger.add_hook(partial(jointplot, loss_type="mse_loss"), feature="val_mse_loss", freq=1)
    logger.add_hook(lambda logger, data: model.save(f"{RESULTS_DIR}/model.pth"), feature="epoch", freq=1)

    # DATA LOADING
    train_loader, val_loader, train_step, val_step, test_set, test_images = load_sintel_train_val_test(src_task, dest_task, batch_size=8) 
    src_task.plot_func(test_images, "images", logger, resize=128)

    for epochs in range(0, 800):

        logger.update("epoch", epochs)
        plot_images(model, logger, test_set, dest_task, show_masks=True)

        train_set = itertools.islice(train_loader, train_step)
        val_set = itertools.islice(val_loader, val_step)

        (train_mse_data,) = model.fit_with_metrics(train_set, loss_fn=dest_task.norm, logger=logger)
        logger.update("train_mse_loss", torch.mean(torch.tensor(train_mse_data)))
        (val_mse_data,) = model.predict_with_metrics(val_set, loss_fn=dest_task.norm, logger=logger)
        logger.update("val_mse_loss", torch.mean(torch.tensor(train_mse_data)))


if __name__ == "__main__":
    Fire(main)
