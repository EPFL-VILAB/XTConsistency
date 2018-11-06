
import numpy as np
import matplotlib as mpl

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
        data_dir=DATA_DIR,
        source_task="rgb",
        dest_task="normal",
        source_transforms=transforms.ToTensor(),
        dest_transforms=transforms.ToTensor(),
    ):
        self.data_dir = data_dir
        self.source_task, self.dest_task, self.buildings = (source_task, dest_task, buildings)
        self.source_transforms, self.dest_transforms = (source_transforms, dest_transforms)
        self.source_files = [
            sorted(glob.glob(f"{data_dir}/{building}_{source_task}/{source_task}/*.png")) for building in buildings
        ]
        self.source_files = []
        for building in buildings:
            self.source_files += sorted(glob.glob(f"{data_dir}/{building}_{source_task}/{source_task}/*.png"))
        # self.source_files = [y for x in self.source_files for y in x]
        print ("Source files len: ", len(self.source_files))

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):

        source_file = self.source_files[idx]
        result = parse.parse(self.data_dir + "/{building}_{task}/{task}/{view}_domain_{task2}.png", source_file)
        data_dir, building, task, view = (self.data_dir, result["building"], result["task"], result["view"])
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



class ImageMultiTaskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        buildings,
        data_dir=DATA_DIR,
        source_task="rgb",
        dest_task=["normal", "principal_curvature"],
        source_transforms=transforms.ToTensor(),
        dest_transforms=transforms.ToTensor(),
    ):
        self.data_dir = data_dir
        self.source_task, self.dest_tasks, self.buildings = (source_task, dest_task, buildings)
        self.source_transforms, self.dest_transforms = (source_transforms, dest_transforms)
        self.source_files = [
            sorted(glob.glob(f"{data_dir}/{building}_{source_task}/{source_task}/*.png")) for building in buildings
        ]
        self.source_files = []
        for building in buildings:
            self.source_files += sorted(glob.glob(f"{data_dir}/{building}_{source_task}/{source_task}/*.png"))
        # self.source_files = [y for x in self.source_files for y in x]
        print ("Source files len: ", len(self.source_files))

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):

        source_file = self.source_files[idx]
        result = parse.parse(self.data_dir + "/{building}_{task}/{task}/{view}_domain_{task2}.png", source_file)
        data_dir, building, task, view = (self.data_dir, result["building"], result["task"], result["view"])

        try:
            image = Image.open(source_file)
            image = self.source_transforms(image).float()

            task_data = []
            for task2 in self.dest_tasks:
                dest_file = f"{data_dir}/{building}_{task2}/{task2}/{view}_domain_{task2}.png"
                task = Image.open(dest_file)
                task = self.dest_transforms(task).float()
                task_data.append(task)
        
        except Exception as e:
            # print (e)
            return self.__getitem__(random.randrange(0, len(self.source_files)))
        
        return image, tuple(task_data)




class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        data_dir=f"{DATA_DIR}/ood_images",
        resize=(256, 256),
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

    test_buildings = ["almena", "albertville"]
    buildings = [file.split("/")[-1][:-7] for file in glob.glob(f"{DATA_DIR}/*_normal")]
    train_buildings, val_buildings = train_test_split(buildings, test_size=0.1)

    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    data_loader = torch.utils.data.DataLoader(
        ImageMultiTaskDataset(buildings=train_buildings, source_transforms=transform, dest_transforms=transform),
        batch_size=64,
        num_workers=0,
        shuffle=True,
    )
    logger.add_hook(lambda data: logger.step(), freq=32)

    for i, (X, Y) in enumerate(data_loader):
        logger.update("epoch", i)
        
