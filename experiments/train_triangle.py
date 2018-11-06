
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

import IPython


def main():

    # MODEL

    curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
    def curvature_model(pred):
        return checkpoint(curvature_model_base, pred)

    depth_model_base = DataParallelModel.load(UNetDepth().cuda(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth")
    def depth_model(pred):
        return checkpoint(depth_model_base, pred)

    class Network(TrainableModel):

        def __init__(self):
            super().__init__()
            self.model = DataParallelModel(UNet())

        def forward(self, x):
            return self.model(depth_model(x).expand(-1, 3, -1, -1))

        def loss(self, pred, target):
            loss = torch.tensor(0.0, device=pred.device)
            return loss, (loss.detach(),)

    model = Network()
    model.compile(torch.optim.Adam, lr=5e-4, weight_decay=2e-6, amsgrad=True)

    def loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), curvature_model(target)*mask.float())
        return mse, (mse.detach(),)

    print (model(torch.randn(8, 3, 256, 256)).shape)
    print (model(torch.randn(16, 3, 256, 256)).shape)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    def jointplot(data):
        data = np.stack((logger.data["train_mse_loss"], logger.data["val_mse_loss"]), axis=1)
        logger.plot(data, "mse_loss", opts={"legend": ["train_mse", "val_mse"]})

    logger.add_hook(jointplot, feature="val_mse_loss", freq=1)
    logger.add_hook(lambda x: model.model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    # DATA LOADING
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("normal", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    plot_images(model, logger, test_set, mask_val=0.502, loss_models={'curvature_target': curvature_model})

    # TRAINING
    for epochs in range(0, 800):

        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (train_mse_data,) = model.fit_with_metrics(train_set, loss_fn=loss, logger=logger)

        logger.update("train_mse_loss", np.mean(train_mse_data))

        val_set = itertools.islice(val_loader, val_step)
        (val_mse_data,) = model.predict_with_metrics(val_set, loss_fn=loss, logger=logger)
        logger.update("val_mse_loss", np.mean(val_mse_data))

        plot_images(model, logger, test_set, mask_val=0.502, loss_models={'curvature': curvature_model})


if __name__ == "__main__":
    Fire(main)