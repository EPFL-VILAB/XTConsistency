
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset
from torch.optim.lr_scheduler import MultiStepLR

from modules.percep_nets import DenseNet, DeepNet, BaseNet, ResidualsNet

import IPython


if __name__ == "__main__":

    # MODEL
    model = DataParallelModel(ResidualsNet())
    model.compile(torch.optim.Adam, lr=2e-4, weight_decay=2e-6, amsgrad=True)
    print (model.forward(torch.randn(1, 3, 512, 512)).shape)

    # LOGGING
    logger = VisdomLogger("train", server='35.230.67.129', port=7000, env=JOB)
    logger.add_hook(lambda x: logger.step(), feature='loss', freq=25)

    def jointplot(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    logger.add_hook(jointplot, feature='val_loss', freq=1)
    logger.add_hook(lambda x: model.save("/result/model.pth"), feature='loss', freq=400)

    train_loader, val_loader, test_set, test_images, ood_images = load_data("normal", "principal_curvature", batch_size=64)
    plot_images(model, logger, test_set, mask_val=0.0)

    # TRAINING
    for epochs in range(0, 200):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, 200)
        (losses,) = model.fit_with_metrics(train_set, logger=logger)
        logger.update('train_loss', np.mean(losses))

        val_set = itertools.islice(val_loader, 200)
        (losses,) = model.predict_with_metrics(val_set, logger=logger)
        logger.update('val_loss', np.mean(losses))

        plot_images(model, logger, test_set, mask_val=0.0)
