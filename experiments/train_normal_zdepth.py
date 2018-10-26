
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

from modules.depth_nets import UNetDepth, ResNetDepth, ResidualsNet
from sklearn.model_selection import train_test_split

import IPython



if __name__ == "__main__":

    # MODEL
    # model = DataParallelModel(ResidualsNet())
    # model = DataParallelModel(ResNetDepth())
    model = DataParallelModel(UNetDepth())

    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    print (model.forward(torch.randn(1, 3, 512, 512)).shape)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature='loss', freq=25)

    def jointplot(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    logger.add_hook(jointplot, feature='val_loss', freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature='loss', freq=400)

     # DATA LOADING
    buildings = [file.split("/")[-1][:-7] for file in glob.glob(f"{DATA_DIR}/*_normal")]
    train_buildings, val_buildings = train_test_split(buildings, test_size=0.1)

    to_tensor = transforms.ToTensor()
    def dest_transforms(x):
        x = to_tensor(x).float().unsqueeze(0)
        mask = build_mask(x, 65535.0, tol=1000)
        x[~mask] = 8000.0
        x = x/8000.0
        return x[0]

    train_loader = torch.utils.data.DataLoader(
                            ImageTaskDataset(buildings=train_buildings, 
                                source_task='normal', dest_task='depth_zbuffer',
                                dest_transforms=dest_transforms),
                        batch_size=64, num_workers=16, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
                            ImageTaskDataset(buildings=val_buildings, 
                                source_task='normal', dest_task='depth_zbuffer',
                                dest_transforms=dest_transforms),
                        batch_size=64, num_workers=16, shuffle=True)

    train_loader, val_loader = cycle(train_loader), cycle(val_loader)
    test_set = list(itertools.islice(val_loader, 1))
    
    plot_images(model, logger, test_set, mask_val=1.0)

    # TRAINING
    for epochs in range(0, 800):
        
        logger.update('epoch', epochs)
        
        train_set = itertools.islice(train_loader, 200)
        (losses,) = model.fit_with_metrics(train_set, logger=logger)
        logger.update('train_loss', np.mean(losses))

        val_set = itertools.islice(val_loader, 200)
        (losses,) = model.predict_with_metrics(val_set, logger=logger)
        logger.update('val_loss', np.mean(losses))

        plot_images(model, logger, test_set, mask_val=1.0)
