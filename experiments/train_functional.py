
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
from functional import get_functional_loss
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImagePairDataset, load_train_val, load_test, load_ood

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from fire import Fire

from functools import partial

from transfers import TRANSFER_MAP

import IPython


def main(loss_config="gt_mse", mode="standard", pretrained=False, **kwargs):

    # MODEL
    model = TRANSFER_MAP['n'].load_model() if pretrained else DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # FUNCTIONAL LOSS
    functional = get_functional_loss(config=loss_config, mode=mode, model=model, **kwargs)
    print(functional)
    print ("Losses: ", functional.losses.keys())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)
    logger.add_hook(lambda x: scheduler.step(), feature="epoch", freq=1)
    functional.logger_hooks(logger)

    # DATA LOADING
    train_loader, val_loader, train_step, val_step = load_train_val("rgb", "normal", batch_size=48)
        # train_buildings=['almena'], val_buildings=['almena'])
    test_set, test_images = load_test("rgb", "normal")
    ood_images = load_ood()
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    # TRAINING
    for epochs in range(0, 50):
        preds_name = "start_preds" if epochs == 0 else "preds"
        ood_name = "start_ood" if epochs == 0 else "ood"
        plot_images(model, logger, test_set, dest_task="normal", ood_images=ood_images, 
            loss_models=functional.plot_losses, preds_name=preds_name, ood_name=ood_name
        )
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        val_set = itertools.islice(val_loader, val_step)

        train_metrics = model.fit_with_metrics(train_set, loss_fn=functional, logger=logger)
        val_metrics = model.predict_with_metrics(val_set, loss_fn=functional, logger=logger)
        functional.logger_update(logger, train_metrics, val_metrics)


if __name__ == "__main__":
    Fire(main)
