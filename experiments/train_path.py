
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

from transfers import TRANSFER_MAP
from task_configs import TASK_MAP

def parse_path(path_str):
    return path_str.replace(')', '').split('(')[::-1][1:]

def main(path_str, src_task='rgb'):

    logger = VisdomLogger("train", env=JOB)

    path = [TRANSFER_MAP[name] for name in parse_path(path_str)]
    root_task = TASK_MAP[src_task]

    prefix = 'x'
    for i, transfer in enumerate(path):
        prefix = f'{transfer.name}({prefix})'
        if i == 0:
            print(f"{transfer.name} is root, skipping")
            continue
        if path[i-1].name == "a":
            print(f"{transfer.name} conditioned on a is not useful, skipping")
            continue

        print(f'training {transfer.name} given {prefix}')
        class PathModel(TrainableModel):
            def __init__(self, transfer):
                super().__init__()
                transfer.load_model()
                self.model = transfer.model
            def forward(self, x):
                with torch.no_grad():
                    for f in path[:i]:
                        x = f(x)
                return self.model(x)
            def loss(self, pred, target):
                loss = torch.tensor(0.0, device=pred.device)
                return loss, (loss.detach(),)

        model = PathModel(path[i])
        model.compile(torch.optim.Adam, lr=1e-4, weight_decay=2e-6, amsgrad=True)
        dest_task = transfer.dest_task

        # LOGGING
        logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

        def jointplot(data):
            data = np.stack((logger.data[f"train_mse_loss_{prefix}"], logger.data[f"val_mse_loss_{prefix}"]), axis=1)
            logger.plot(data, f"mse_loss_{prefix}", opts={"legend": ["train_mse", "val_mse"]})

        logger.add_hook(jointplot, feature=f"val_mse_loss_{prefix}", freq=1)
        logger.add_hook(lambda x: model.model.save(get_finetuned_model_path(path[:i+1])), feature="loss", freq=400)

        # DATA LOADING
        train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
            load_data(root_task, dest_task, batch_size=64)
        logger.images(test_images, "images", resize=128)

        plot_images(model, logger, test_set, mask_val=dest_task.mask_val, target_name=f"targets_{prefix}", preds_name=f"preds_{prefix}_start")

        # TRAINING
        for epochs in range(0, 20):

            logger.update("epoch", epochs)

            train_set = itertools.islice(train_loader, train_step)
            (train_mse_data,) = model.fit_with_metrics(train_set, loss_fn=dest_task.loss_func, logger=logger)
            logger.update(f"train_mse_loss_{prefix}", np.mean(train_mse_data))

            val_set = itertools.islice(val_loader, val_step)
            (val_mse_data,) = model.predict_with_metrics(val_set, loss_fn=dest_task.loss_func, logger=logger)
            logger.update(f"val_mse_loss_{prefix}", np.mean(val_mse_data))
            if epochs == 0:
                print('starting mse: ', val_mse_data[0])


            plot_images(model, logger, test_set, mask_val=dest_task.mask_val, target_name=f"targets_{prefix}", preds_name=f"preds_{prefix}")
            # val_mse_arr.extend(list(val))


if __name__ == "__main__":
    Fire(main)
