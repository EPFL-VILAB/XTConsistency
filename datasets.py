
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
from sklearn.model_selection import train_test_split
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

""" Default data loading configurations for training, validation, and testing. """
def load_sintel_train_val_test(source_task, dest_task, 
        batch_size=64, batch_transforms=cycle
    ):

    if isinstance(source_task, str) and isinstance(dest_task, str):
        source_task, dest_task = get_task(source_task), get_task(dest_task)
    
    buildings = sorted([x.split('/')[-1] for x in glob.glob("mount/sintel/training/depth/*")])
    train_buildings, val_buildings = train_test_split(buildings, test_size=0.2)
    print (len(train_buildings))
    print (len(val_buildings))

    train_loader = torch.utils.data.DataLoader(
        SintelDataset(buildings=train_buildings, tasks=[source_task, dest_task]),
        batch_size=batch_size,
        num_workers=64, shuffle=True, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        SintelDataset(buildings=val_buildings, tasks=[source_task, dest_task]),
        batch_size=batch_size,
        num_workers=64, shuffle=True, pin_memory=True
    )

    train_step = int(2248616 // (100 * batch_size))
    val_step = int(245592 // (100 * batch_size))
    print("Train step: ", train_step)
    print("Val step: ", val_step)

    test_set = list(itertools.islice(val_loader, 1))
    test_images = torch.cat([x for x, y in test_set], dim=0)
    
    return train_loader, val_loader, train_step, val_step, test_set, test_images


""" Load all buildings """
def load_all(tasks, buildings=None, batch_size=64, split_file="data/split.txt", batch_transforms=cycle):

    data = yaml.load(open(split_file))
    buildings = buildings or (data["train_buildings"] + data["val_buildings"])

    data_loader = torch.utils.data.DataLoader(
        TaskDataset(buildings=buildings, tasks=tasks),
        batch_size=batch_size,
        num_workers=0, shuffle=True, pin_memory=True
    )

    return data_loader



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

def load_test(source_task, dest_task, sample=32):
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


def load_ood(ood_path=f"{SHARED_DIR}/ood_standard_set", resize=256):
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
    




class TaskDataset(Dataset):

    def __init__(self, buildings, tasks=[get_task("rgb"), get_task("normal")], data_dirs=DATA_DIRS, 
            building_files=None, convert_path=None, use_raid=USE_RAID):

        super().__init__()
        self.buildings, self.tasks, self.data_dirs = buildings, tasks, data_dirs
        self.building_files = building_files or self.building_files
        self.convert_path = convert_path or self.convert_path
        if use_raid:
            self.convert_path = self.convert_path_raid
            self.building_files = self.building_files_raid
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
                task_files += sorted(self.building_files(task, building))
            print(f"{task.name} file len: {len(task_files)}")
            task_set = {self.convert_path(x, tasks[0]) for x in task_files}
            filtered_files = filtered_files.intersection(task_set) if i != 0 else task_set

        self.idx_files = sorted(list(filtered_files))
        print ("Intersection files len: ", len(self.idx_files))

    def building_files(self, task, building):
        """ Gets all the tasks in a given building (grouping of data) """
        return get_files(f"{building}_{task.file_name}/{task.file_name}/*.{task.file_ext}", self.data_dirs)
    def building_files_raid(self, task, building):
        return get_files(f"{task}/{building}/*.{task.file_ext}", self.data_dirs)
    def convert_path(self, source_file, task):
        """ Converts a file from task A to task B. Can be overriden by subclasses"""
        source_file = "/".join(source_file.split('/')[-3:])
        result = parse.parse("{building}_{task}/{task}/{view}_domain_{task2}.{ext}", source_file)
        building, _, view = (result["building"], result["task"], result["view"])
        dest_file = f"{building}_{task.file_name}/{task.file_name}/{view}_domain_{task.file_name_alt}.{task.file_ext}"
        if f"{building}_{task.file_name}" not in self.file_map:
            print (f"{building}_{task.file_name} not in file map")
            return ""
        data_dir = self.file_map[f"{building}_{task.file_name}"]
        return f"{data_dir}/{dest_file}"

    def convert_path_raid(self, full_file, task):
        """ Converts a file from task A to task B. Can be overriden by subclasses"""
        source_file = "/".join(full_file.split('/')[-3:])
        result = parse.parse("{task}/{building}/{view}.{ext}", source_file)
        building, _, view = (result["building"], result["task"], result["view"])
        dest_file = f"{task}/{building}/{view}.{task.file_ext}"
        return f"{full_file[:-len(source_file)-1]}/{dest_file}"

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


class SintelDataset(TaskDataset):

    def __init__(self, *args, **kwargs):
        if "buildings" not in kwargs or ["buildings"] is None:
            kwargs["buildings"] = sorted([x.split('/')[-1] for x in glob.glob("mount/sintel/training/depth/*")])
        super().__init__(*args, **kwargs)

    def building_files(self, task, building):
        """ Gets all the tasks in a given building (grouping of data) """
        task_dir = {"rgb": "clean", "normal": "depth", "depth_zbuffer": "depth"}[task.name]
        task_val = {"rgb": "frame", "normal": "normal", "depth_zbuffer": "frame"}[task.name]

        return sorted(glob.glob(f"mount/sintel/training/{task_dir}/{building}/{task_val}*.png"))

    def convert_path(self, source_file, task):
        """ Converts a file from task A to task B. Can be overriden by subclasses"""
        result = parse.parse("mount/sintel/training/{task_dir}/{building}/{task_val}_{view}.png", source_file)
        building, view = (result["building"], result["view"])

        task_dir = {"rgb": "clean", "normal": "depth", "depth_zbuffer": "depth"}[task.name]
        task_val = {"rgb": "frame", "normal": "normal", "depth_zbuffer": "frame"}[task.name]

        dest_file = f"mount/sintel/training/{task_dir}/{building}/{task_val}_{view}.png"
        return dest_file
            


class ImageDataset(Dataset):

    def __init__(
        self,
        data_dir=f"data/ood_images",
        resize=(256, 256),
    ):
        def crop(x):
            return transforms.CenterCrop(min(x.size[0], x.size[1]))(x)
        self.transforms = transforms.Compose([crop, transforms.Resize(resize), transforms.ToTensor()])
        os.system(f"ls {data_dir}/*.png")
        os.system(f"sudo ls {data_dir}/*.png")
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
    train_loader = TaskDataset(buildings=["almena", "albertville"], tasks=[tasks.rgb, tasks.normal])
    logger.add_hook(lambda logger, data: logger.step(), freq=32)

    for i, (X, Y) in enumerate(train_loader):
        logger.update("epoch", i)
