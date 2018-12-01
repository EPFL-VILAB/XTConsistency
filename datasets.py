
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
from task_configs import get_task, tasks

from PIL import Image
from io import BytesIO
import IPython


""" Default data loading configurations for training, validation, and testing. """
def load_train_val(source_task, dest_task, 
        train_buildings=None, val_buildings=None, split_file="data/split.txt", 
        batch_size=64, batch_transforms=cycle
    ):

    if isinstance(source_task, str) and isinstance(dest_task, str):
        source_task, dest_task = get_task(source_task), get_task(dest_task)
    
    data = yaml.load(open(split_file))
    train_buildings = train_buildings or data["train_buildings"]
    val_buildings = val_buildings or data["val_buildings"]

    train_loader = torch.utils.data.DataLoader(
        TaskDataset(buildings=train_buildings, tasks=[source_task, dest_task]),
        batch_size=batch_size,
        num_workers=64, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        TaskDataset(buildings=val_buildings, tasks=[source_task, dest_task]),
        batch_size=batch_size,
        num_workers=64, shuffle=True, pin_memory=True
    )

    train_step = int(2248616 // (100 * batch_size))
    val_step = int(245592 // (100 * batch_size))
    print("Train step: ", train_step)
    print("Val step: ", val_step)

    return train_loader, val_loader, train_step, val_step


def load_test(source_task, dest_task, 
        buildings=["almena", "albertville"], sample=6,
    ):
    if isinstance(source_task, str) and isinstance(dest_task, str):
        source_task, dest_task = get_task(source_task), get_task(dest_task)

    test_loader1 = torch.utils.data.DataLoader(
        TaskDataset(buildings=[buildings[0]], tasks=[source_task, dest_task]),
        batch_size=sample,
        num_workers=sample, shuffle=False, pin_memory=True
    )
    test_loader2 = torch.utils.data.DataLoader(
        TaskDataset(buildings=[buildings[1]], tasks=[source_task, dest_task]),
        batch_size=sample,
        num_workers=sample, shuffle=False, pin_memory=True
    )
    test_set = list(itertools.islice(test_loader1, 1)) + list(itertools.islice(test_loader2, 1))
    test_images = torch.cat([x for x, y in test_set], dim=0)

    return test_set, test_images


def load_ood(ood_path=f"{SHARED_DIR}/ood_standard_set/", resize=256):
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir=ood_path, resize=(resize, resize)),
        batch_size=10,
        num_workers=10, shuffle=False, pin_memory=True
    )
    ood_images = list(itertools.islice(ood_loader, 1))
    return ood_images


def load_video_games(ood_path=f"{BASE_DIR}/video_games", resize=256):
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir=ood_path, resize=(resize, resize)),
        batch_size=66,
        num_workers=32, shuffle=False, pin_memory=True
    )
    ood_images = list(itertools.islice(ood_loader, 1))
    return ood_images


def load_doom(ood_path=f"{BASE_DIR}/Doom/video2", resize=256):
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir=ood_path, resize=(resize, resize)),
        batch_size=64,
        num_workers=64, shuffle=False, pin_memory=True
    )
    ood_images = list(itertools.islice(ood_loader, 1))
    return ood_loader, ood_images
    

def load_sintel_train_val_test(batch_size=32):
    train_loader = torch.utils.data.DataLoader(
        ImagePairDataset(data_dir=f"{BASE_DIR}/sintel/training/clean", 
            files=glob.glob(f"{BASE_DIR}/sintel/training/clean/*/*.png")
        ),
        batch_size=batch_size,
        num_workers=batch_size, shuffle=False, pin_memory=True
    )
    test_files = glob.glob(f"{BASE_DIR}/sintel/test/clean/*/*.png")
    random.Random(229).shuffle(test_files)
    val_loader = torch.utils.data.DataLoader(
        ImagePairDataset(data_dir=f"{BASE_DIR}/sintel/test/clean", 
            files=test_files
        ),
        batch_size=batch_size,
        num_workers=batch_size, shuffle=False, pin_memory=True
    )
    test_set = list(itertools.islice(val_loader, 1))
    test_images = torch.cat([x for x, y in test_set], dim=0)

    return train_loader, val_loader, test_set, test_images




class TaskDataset(Dataset):

    def __init__(self, buildings, tasks=[get_task("rgb"), get_task("normal")], data_dirs=DATA_DIRS):

        super().__init__()
        self.buildings, self.tasks, self.data_dirs = buildings, tasks, data_dirs

        # Build a map from buildings to directories
        self.file_map = {}
        for data_dir in self.data_dirs:
            for file in glob.glob(f'{data_dir}/*'):
                res = parse.parse("{building}_{task}", file[len(data_dir)+1:])
                if res is None: continue
                self.file_map[file[len(data_dir)+1:]] = data_dir

        filtered_files = set()
        for i, task in enumerate(tasks):
            task_files = []
            for building in buildings:
                task_files += sorted(get_files(f"{building}_{task.file_name}/{task.file_name}/*.{task.file_ext}", data_dirs))
            print(f"{task.name} file len: {len(task_files)}")
            task_set = {self.convert_path(x, tasks[0]) for x in task_files}
            filtered_files = filtered_files.intersection(task_set) if i != 0 else task_set

        self.idx_files = sorted(list(filtered_files))
        print ("Intersection files len: ", len(self.idx_files))

    def convert_path(self, source_file, task):
        """ Converts a file from task A to task B. """

        result = parse.parse("{building}_{task}/{task}/{view}_domain_{task2}.{ext}", "/".join(source_file.split('/')[-3:]))
        building, _, view = (result["building"], result["task"], result["view"])
        dest_file = f"{building}_{task.file_name}/{task.file_name}/{view}_domain_{task.file_name_alt}.{task.file_ext}"
        if f"{building}_{task.file_name}" not in self.file_map:
            return ""
        data_dir = self.file_map[f"{building}_{task.file_name}"]
        return f"{data_dir}/{dest_file}"

    def __len__(self):
        return len(self.idx_files)

    def __getitem__(self, idx):

        for i in range(200):
            try:
                res = []
                for task in self.tasks:
                    file_name = self.convert_path(self.idx_files[idx], task)
                    if len(file_name) == 0: raise Exception("unable to convert file")
                    image = task.file_loader(file_name)
                    res.append(image)
                return tuple(res)
            except Exception as e:
                idx = random.randrange(0, len(self.idx_files))
                if i == 199: raise (e)
            


class ImageDataset(Dataset):

    def __init__(
        self,
        data_dir=f"data/ood_images",
        resize=(256, 256),
    ):
        def crop(x):
            return transforms.CenterCrop(min(x.size[0], x.size[1]))(x)
        self.transforms = transforms.Compose([crop, transforms.Resize(resize), transforms.ToTensor()])
        self.files = glob.glob(f"{data_dir}/*.png") + glob.glob(f"{data_dir}/*.jpg") + glob.glob(f"{data_dir}/*.jpeg")
        self.files = sorted(self.files)
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
        return image


class ImagePairDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, data_dir, resize=(256, 256), files=None):

        self.data_dir = data_dir
        def crop(x):
            return transforms.CenterCrop(min(x.size[0], x.size[1]))(x)
        self.transforms = transforms.Compose([crop, transforms.Resize(resize), transforms.ToTensor()])

        self.files = files or glob.glob(f"{data_dir}/*.png") + glob.glob(f"{data_dir}/*.jpg") + glob.glob(f"{data_dir}/*.jpeg")
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
        return image, image




if __name__ == "__main__":

    logger = VisdomLogger("data", env=JOB)

    train_loader, val_loader, test_set, test_images = load_sintel_train_val_test()
    logger.images(test_images, "test_images", resize=256)

    logger.add_hook(lambda data: logger.step(), freq=32)

    for i, (X, Y) in enumerate(train_loader):
        logger.update("epoch", i)
        
