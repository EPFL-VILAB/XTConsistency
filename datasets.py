
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import os, sys, math, random, tarfile, glob, time, itertools
import parse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import *
from logger import Logger, VisdomLogger

from PIL import Image
from io import BytesIO
import IPython


class ImageTaskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        buildings,
        data_dir="/data",
        source_task="rgb",
        dest_task="normal",
        source_transforms=transforms.ToTensor(),
        dest_transforms=transforms.ToTensor(),
    ):

        self.source_task, self.dest_task, self.buildings = (source_task, dest_task, buildings)
        self.source_transforms, self.dest_transforms = (source_transforms, dest_transforms)
        self.source_files = [
            sorted(glob.glob(f"{data_dir}/{building}_{source_task}/{source_task}/*.png")) for building in buildings
        ]
        self.source_files = [y for x in self.source_files for y in x]

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):

        source_file = self.source_files[idx]

        result = parse.parse("{data_dir}/{building}_{task}/{task}/{view}_domain_{task}.png", source_file)
        data_dir, building, task, view = (result["data_dir"], result["building"], result["task"], result["view"])
        dest_file = f"{data_dir}/{building}_{self.dest_task}/{self.dest_task}/{view}_domain_{self.dest_task}.png"

        try:
            image = Image.open(source_file)
            image = self.source_transforms(image).float()

            task = Image.open(dest_file)
            task = self.dest_transforms(task).float()
            return image, task
        except:
            # print ("Error in file pair: ", source_file, dest_file)
            time.sleep(0.1)
            return self.__getitem__(random.randrange(0, len(self.source_files)))


class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        data_dir="/data/ood_images",
        resize=(512, 512),
    ):

        self.transforms = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
        self.files = glob.glob(f"{data_dir}/*.png")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        image = Image.open(file)
        image = self.transforms(image).float()[0:3, :, :]
        return image


if __name__ == "__main__":

    logger = VisdomLogger("data", server="35.230.67.129", port=7000, env=JOB)

    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir="data/ood_images"),
        batch_size=6,
        num_workers=6,
        shuffle=False,
    )
    ood_images = torch.cat(list(itertools.islice(ood_loader, 1)), dim=0)
    logger.images(ood_images, "ood_images", resize=128)

    # dataset = ImageTaskDataset(
    #     buildings=["ackermanville", "adairsville", "adrian", "airport", "akiak"], data_dir="result"
    # )
    # data_loader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=32, shuffle=True)
    # logger = Logger("data")
    # logger.add_hook(lambda data: logger.step(), freq=16)

    # for i, (X, Y) in enumerate(data_loader):
    #     logger.update("epoch", i)
