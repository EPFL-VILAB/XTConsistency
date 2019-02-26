
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
from datasets import load_train_val, load_test, load_ood

from fire import Fire
import IPython


def main():

    src_task, dest_task = tasks.rgb, tasks.normal
    model = DataParallelModel(get_model(src_task, dest_task).cuda())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=25)
    logger.add_hook(partial(jointplot, loss_type="consistency"), feature="val_consistency", freq=1)
    logger.add_hook(partial(jointplot, loss_type="correctness"), feature="val_correctness", freq=1)
    logger.add_hook(lambda logger, data: model.save(f"{RESULTS_DIR}/{src_task.name}2{dest_task.name}.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, train_step, val_step = load_train_val([tasks.rgb, tasks.principal_curvature], batch_size=32,
        # train_buildings=["almena"], val_buildings=["almena"]
    )
    train_loader2, val_loader2, _, _ = load_train_val([tasks.rgb, tasks.normal], batch_size=32,
        train_buildings=["almena"], val_buildings=["almena"])
    test_set, test_images = load_test(src_task, dest_task)
    src_task.plot_func(test_images, "images", logger, resize=128)

    scale = train_step / 4
    train_step, val_step = 4, 4

    for epochs in range(0, 4000):

        logger.update("epoch", epochs*1.0/scale)
        plot_images(model, logger, test_set, dest_task, show_masks=True)

        train_set = itertools.islice([train_loader, train_loader2][epochs % 2], train_step)
        val_set = itertools.islice([val_loader, val_loader2][epochs % 2], val_step)

        def loss_fn(y, y_hat):
            return tasks.principal_curvature.norm(functional_transfers.f(y), y_hat)

        def loss_fn2(y, y_hat):
            return tasks.normal.norm(y, y_hat)

        loss_fn = [loss_fn, loss_fn2][epochs % 2]

        train_mse_data = model.fit_with_metrics(train_set, loss_fn=loss_fn, logger=logger)
        val_mse_data = model.predict_with_metrics(val_set, loss_fn=loss_fn, logger=logger)
        name = ["consistency", "correctness"] [epochs % 2]

        if epochs % 17 == 0:
            logger.update(f"train_{name}", np.mean(train_mse_data))
            logger.update(f"val_{name}", np.mean(val_mse_data))


if __name__ == "__main__":
    Fire(main)
