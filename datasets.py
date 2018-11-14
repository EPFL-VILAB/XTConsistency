
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
        data_dirs=DATA_DIRS,
        source_task=None,
        dest_task=None,
        source_transforms=transforms.ToTensor(),
        dest_transforms=transforms.ToTensor(),
    ):
        self.data_dirs = data_dirs
        self.source_task, self.dest_task, self.buildings = (source_task, dest_task, buildings)
        self.source_transforms, self.dest_transforms = (source_transforms, dest_transforms)

        self.source_files = []
        for building in buildings:
            self.source_files += sorted(get_files(f"{building}_{source_task.file_name()}/{source_task.file_name()}/*.png", data_dirs))
            
        print ("Source files len: ", len(self.source_files))
        target_files = []
        for building in buildings:
            target_files += sorted(get_files(f"{building}_{dest_task.file_name()}/{dest_task.file_name()}/*.png", data_dirs))
        print("Target files len: ", len(target_files))
        target_files_set = {convert_path(x, source_task.file_name()) for x in target_files}
        source_files_set = set(self.source_files)
        source_files_set = source_files_set.intersection(target_files_set)
        self.source_files = list(source_files_set)
        print("Filtered files len: ", len(self.source_files))

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):
        for i in range(20):
            source_file = self.source_files[idx]
            dest_file = convert_path(source_file, self.dest_task.file_name())
            try:
                image = self.source_task.file_loader(source_file)
                image = self.source_transforms(image).float()
                task = self.dest_task.file_loader(dest_file)
                task = self.dest_transforms(task).float()
                return image, task
            except Exception as e:
                # time.sleep(0.1)
                # print(i, e)
                idx = random.randrange(0, len(self.source_files))
                # return self.__getitem__(random.randrange(0, len(self.source_files)))
        print('error, error, could not open file')
        return (None, None)

class ImageMultiTaskDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        buildings,
        data_dirs=DATA_DIRS,
        source_task="rgb",
        dest_task=["normal", "principal_curvature"],
        source_transforms=transforms.ToTensor(),
        dest_transforms=transforms.ToTensor(),
    ):
        self.data_dirs = data_dirs
        self.source_task, self.dest_tasks, self.buildings = (source_task, dest_task, buildings)
        self.source_transforms, self.dest_transforms = (source_transforms, dest_transforms)
        # self.source_files = [
        #     sorted(glob.glob(f"{data_dir}/{building}_{source_task}/{source_task}/*.png")) for building in buildings
        # ]
        self.source_files = []
        for building in buildings:
            self.source_files += sorted(get_files(f"{building}_{source_task}/{source_task}/*.png", data_dirs))
        # self.source_files = [y for x in self.source_files for y in x]
        print ("Source files len: ", len(self.source_files))

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, idx):

        source_file = self.source_files[idx]
        result = parse.parse("{building}_{task}/{task}/{view}_domain_{task2}.png", "/".join(source_file.split('/')[-3:]))
        building, task, view = (result["building"], result["task"], result["view"])

        try:
            image = self.Image.open(source_file)
            image = self.source_transforms(image).float()

            task_data = []
            for task2 in self.dest_tasks:
                dest_file = f"{building}_{task2}/{task2}/{view}_domain_{task2}.png"
                data_dir = FILE_MAP[f"{building}_{task2}"]
                task = Image.open(f"{data_dir}/{dest_file}")
                task = self.dest_transforms(task).float()
                task_data.append(task)
        
        except Exception as e:
            print (e)
            return self.__getitem__(random.randrange(0, len(self.source_files)))
        
        return image, tuple(task_data)




class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        data_dir=f"data/ood_images",
        resize=(256, 256),
    ):
        def crop(x):
            return transforms.CenterCrop(min(x.size[0], x.size[1]))(x)
        self.transforms = transforms.Compose([crop, transforms.Resize(resize), transforms.ToTensor()])
        self.files = glob.glob(f"{data_dir}/*.png") + glob.glob(f"{data_dir}/*.jpg") + glob.glob(f"{data_dir}/*.jpeg")
        print("num files = ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        try:
            image = Image.open(file)
            image = self.transforms(image).float()[0:3, :, :]
            if image.shape[0] == 1: image = image.expand(3, -1, -1)
        except Exception as e:
            return self.__getitem__(random.randrange(0, len(self.files)))
        # print(image.shape, file)
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
        
