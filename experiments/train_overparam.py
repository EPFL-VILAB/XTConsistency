
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
from datasets import ImagePairDataset, DualSubsetDataset, load_train_val, load_test, load_ood
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

from transfers import functional_transfers
from task_configs import tasks

import IPython


def main(pretrained=False, batch_size=24):

    model = functional_transfers.n.load_model() if pretrained else DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=(3e-5 if pretrained else 3e-4), weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
    logger.add_hook(lambda logger, data: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)
    logger.add_hook(lambda logger, data: scheduler.step(), feature="epoch", freq=1)
    logger.add_hook(partial(jointplot, loss_type="consistency"), feature="val_consistency", freq=1)
    logger.add_hook(partial(jointplot, loss_type="correctness"), feature="val_correctness", freq=1)

    # DATA LOADING
    
    train_loader, val_loader, train_step, val_step = load_train_val([tasks.rgb, tasks.normal], 
        dataset_cls=DualSubsetDataset, batch_size=batch_size,
        # train_buildings=["almena"], val_buildings=["almena"]
    )
    test_set, test_images = load_test("rgb", "normal")
    ood_images = load_ood()
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    # TRAINING
    for epochs in range(0, 800):
        preds_name = "start_preds" if epochs == 0 and pretrained else "preds"
        ood_name = "start_ood" if epochs == 0 and pretrained else "ood"
        
        plot_images(model, logger, test_set, dest_task="normal", ood_images=ood_images, 
            loss_models={
                "f(y)": lambda y, y_hat, x: functional_transfers.f(y),
                "f(y^)": lambda y, y_hat, x: functional_transfers.f(y_hat),
            }, 
            preds_name=preds_name, ood_name=ood_name
        )

        train_consistency, train_correctness = [], []
        model.train(True)
        for X, Y, X2, Y2 in itertools.islice(train_loader, train_step):
            loss1, _ = tasks.principal_curvature.norm(functional_transfers.f(model(X)), functional_transfers.f(Y))
            loss2, _ = tasks.normal.norm(model(X2), Y2.to(DEVICE))
            loss = loss1 + loss2
            model.step(loss)
            logger.update("loss", loss)

            train_consistency.append(loss1.cpu().data.numpy().mean())
            train_correctness.append(loss2.cpu().data.numpy().mean())

        del loss

        val_consistency, val_correctness = [], []
        with torch.no_grad():
            model.train(False)
            for X, Y, X2, Y2 in itertools.islice(val_loader, val_step):
                loss1, _ = tasks.principal_curvature.norm(functional_transfers.f(model(X)), functional_transfers.f(Y))
                loss2, _ = tasks.normal.norm(model(X2), Y2.to(DEVICE))

                val_consistency.append(loss1.cpu().data.numpy().mean())
                val_correctness.append(loss2.cpu().data.numpy().mean())

        logger.update("train_correctness", train_correctness)
        logger.update("val_correctness", val_correctness)
        logger.update("train_consistency", train_consistency)
        logger.update("val_consistency", val_consistency)
        logger.update("epoch", epochs)



if __name__ == "__main__":
    Fire(main)
