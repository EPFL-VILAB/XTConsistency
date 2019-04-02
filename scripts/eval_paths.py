
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from utils import *
from plotting import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import load_doom

from transfers import TRANSFER_MAP
from task_configs import get_task

import IPython
from fire import Fire


def main():

    logger = VisdomLogger("train", env=JOB)

    test_loader, test_images = load_doom()
    test_images = torch.cat(test_images, dim=0)
    src_task, dest_task = get_task("rgb"), get_task("normal")

    print (test_images.shape)
    src_task.plot_func(test_images, f"images", logger, resize=128)

    paths = ["F(RC(x))", "F(EC(a(x)))", "n(x)", "npstep(x)"]

    for path_str in paths:

        path_list = path_str.replace(')', '').split('(')[::-1][1:]
        path = [TRANSFER_MAP[name] for name in path_list]

        class PathModel(TrainableModel):

            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                with torch.no_grad():
                    for f in path:
                        x = f(x)
                return x

            def loss(self, pred, target):
                loss = torch.tensor(0.0, device=pred.device)
                return loss, (loss.detach(),)

        model = PathModel()

        preds = model.predict(test_loader)
        dest_task.plot_func(preds, f"preds_{path_str}", logger, resize=128)
        transform = transforms.ToPILImage()
        os.makedirs(f"{BASE_DIR}/doom_processed/{path_str}/video2", exist_ok=True)

        for image, file in zip(preds, test_loader.dataset.files):
            image = transform(image.cpu())
            filename = file.split("/")[-1]
            print (filename)
            image.save(f"{BASE_DIR}/doom_processed/{path_str}/video2/{filename}")


    print (preds.shape)


if __name__ == "__main__":
    Fire(main)
