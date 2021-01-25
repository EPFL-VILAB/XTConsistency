
import numpy as np
import matplotlib as mpl

import os, sys, math, random, tarfile, glob, time, yaml, itertools
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

import pdb

""" Default data loading configurations for training, validation, and testing. """
def load_train_val(train_tasks, val_tasks=None, fast=False,
        train_buildings=None, val_buildings=None, split_file="config/split.txt",
        dataset_cls=None, batch_size=32, batch_transforms=cycle,
        subset=None, subset_size=None, dataaug=False,
    ):

    dataset_cls = dataset_cls or TaskDataset
    train_cls = TrainTaskDataset if dataaug else dataset_cls
    train_tasks = [get_task(t) if isinstance(t, str) else t for t in train_tasks]
    if val_tasks is None: val_tasks = train_tasks
    val_tasks = [get_task(t) if isinstance(t, str) else t for t in val_tasks]  
    data = yaml.load(open(split_file))
    train_buildings = train_buildings or (["almena"] if fast else data["train_buildings"])
    val_buildings = val_buildings or (["almena"] if fast else data["val_buildings"])
    print("number of train images:")
    train_loader = train_cls(buildings=train_buildings, tasks=train_tasks)
    print("number of val images:")
    val_loader = dataset_cls(buildings=val_buildings, tasks=val_tasks)

    if subset_size is not None or subset is not None:
        train_loader = torch.utils.data.Subset(train_loader,
            random.sample(range(len(train_loader)), subset_size or int(len(train_loader)*subset)),
        )

    train_step = int(len(train_loader) // (400 * batch_size))
    val_step = int(len(val_loader) // (400 * batch_size))
    print("Train step: ", train_step)
    print("Val step: ", val_step)
    if fast: train_step, val_step = 8, 8

    return train_loader, val_loader, train_step, val_step


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



def load_test(all_tasks, buildings=["almena", "albertville"], sample=4):

    all_tasks = [get_task(t) if isinstance(t, str) else t for t in all_tasks]
    print(f"number of images in {buildings[0]}:")
    test_loader1 = torch.utils.data.DataLoader(
        TaskDataset(buildings=[buildings[0]], tasks=all_tasks, shuffle=False),
        batch_size=sample,
        num_workers=0, shuffle=False, pin_memory=True,
    )
    print(f"number of images in {buildings[1]}:")
    test_loader2 = torch.utils.data.DataLoader(
        TaskDataset(buildings=[buildings[1]], tasks=all_tasks, shuffle=False),
        batch_size=sample,
        num_workers=0, shuffle=False, pin_memory=True,
    )
    set1 = list(itertools.islice(test_loader1, 1))[0]
    set2 = list(itertools.islice(test_loader2, 1))[0]
    test_set = tuple(torch.cat([x, y], dim=0) for x, y in zip(set1, set2))
    return test_set


def load_ood(tasks=[tasks.rgb], ood_path=OOD_DIR, sample=21):
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(tasks=tasks, data_dir=ood_path),
        batch_size=sample,
        num_workers=sample, shuffle=False, pin_memory=True
    )
    ood_images = list(itertools.islice(ood_loader, 1))[0]
    return ood_images



class TaskDataset(Dataset):

    def __init__(self, buildings, tasks=[get_task("rgb"), get_task("normal")], data_dirs=DATA_DIRS,
            building_files=None, convert_path=None, use_raid=USE_RAID, resize=None, unpaired=False, shuffle=True):

        super().__init__()
        self.buildings, self.tasks, self.data_dirs = buildings, tasks, data_dirs
        self.building_files = building_files or self.building_files
        self.convert_path = convert_path or self.convert_path
        self.resize = resize
        if use_raid:
            self.convert_path = self.convert_path_raid
            self.building_files = self.building_files_raid

        self.file_map = {}
        for data_dir in self.data_dirs:
            for file in glob.glob(f'{data_dir}/*'):
                res = parse.parse("{building}_{task}", file[len(data_dir)+1:])
                if res is None: continue
                self.file_map[file[len(data_dir)+1:]] = data_dir

        filtered_files = None

        assert (len(tasks) > 0), "Building dataset for tasks, but no tasks specified!"
        task = tasks[0]
        task_files = []
        for building in buildings:
            task_files += self.building_files(task, building)
        print(f"    {task.name} file len: {len(task_files)}")
        self.idx_files = task_files
        if not shuffle: self.idx_files = sorted(task_files)

        print ("    Intersection files len: ", len(self.idx_files))

    def reset_unpaired(self):
        if self.unpaired:
            self.task_indices = {task:random.sample(range(len(self.idx_files)), len(self.idx_files)) for task in self.task_indices}

    def building_files(self, task, building):
        """ Gets all the tasks in a given building (grouping of data) """
        return get_files(f"{building}_{task.file_name}/{task.file_name}/*.{task.file_ext}", self.data_dirs)

    def building_files_raid(self, task, building):
        return get_files(f"{task.file_name}/{building}/*.{task.file_ext}", self.data_dirs)

    def convert_path(self, source_file, task):
        """ Converts a file from task A to task B. Can be overriden by subclasses"""
        source_file = "/".join(source_file.split('/')[-3:])
        result = parse.parse("{building}_{task}/{task}/{view}_domain_{task2}.{ext}", source_file)
        building, _, view = (result["building"], result["task"], result["view"])
        dest_file = f"{building}_{task.file_name}/{task.file_name}/{view}_domain_{task.file_name_alt}.{task.file_ext}"
        if f"{building}_{task.file_name}" not in self.file_map:
            print (f"{building}_{task.file_name} not in file map")
            # IPython.embed()
            return ""
        data_dir = self.file_map[f"{building}_{task.file_name}"]
        return f"{data_dir}/{dest_file}"

    def convert_path_raid(self, full_file, task):
        """ Converts a file from task A to task B. Can be overriden by subclasses"""
        source_file = "/".join(full_file.split('/')[-3:])
        result = parse.parse("{task}/{building}/{view}.{ext}", source_file)
        building, _, view = (result["building"], result["task"], result["view"])
        dest_file = f"{task.file_name}/{building}/{view}.{task.file_ext}"
        return f"{full_file[:-len(source_file)-1]}/{dest_file}"

    def __len__(self):
        return len(self.idx_files)

    def __getitem__(self, idx):

        for i in range(200):
            try:
                res = []

                seed = random.randint(0, 1e10)

                for task in self.tasks:
                    file_name = self.convert_path(self.idx_files[idx], task)
                    if len(file_name) == 0: raise Exception("unable to convert file")
                    image = task.file_loader(file_name, resize=self.resize, seed=seed)

                    res.append(image)
                return tuple(res)
            except Exception as e:
                idx = random.randrange(0, len(self.idx_files))
                if i == 199: raise (e)


class TrainTaskDataset(TaskDataset):

    def __getitem__(self, idx):

        for i in range(200):
            try:
                res = []

                seed = random.randint(0, 1e10)
                crop = random.randint(int(0.7*512), 512) if bool(random.getrandbits(1)) else 512

                for task in self.tasks:
                    jitter = bool(random.getrandbits(1)) if task.name == 'rgb' else False
                    file_name = self.convert_path(self.idx_files[idx], task)
                    if len(file_name) == 0: raise Exception("unable to convert file")
                    image = task.file_loader(file_name, resize=self.resize, seed=seed, crop=crop, jitter=jitter)
                    res.append(image)

                return tuple(res)
            except Exception as e:
                idx = random.randrange(0, len(self.idx_files))
                if i == 199: raise (e)


class ImageDataset(Dataset):

    def __init__(
        self,
        tasks=[tasks.rgb],
        data_dir=f"data/ood_images",
        files=None,
    ):

        self.tasks = tasks
        #if not USE_RAID and files is None:
        #    os.system(f"ls {data_dir}/*.png")
        #    os.system(f"ls {data_dir}/*.png")

        self.files = files \
            or sorted(
                glob.glob(f"{data_dir}/*.png")
                + glob.glob(f"{data_dir}/*.jpg")
                + glob.glob(f"{data_dir}/*.jpeg")
            )

        print("number of ood images: ", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        res = []
        seed = random.randint(0, 1e10)
        for task in self.tasks:
            image = task.file_loader(file, seed=seed)
            if image.shape[0] == 1: image = image.expand(3, -1, -1)
            res.append(image)
        return tuple(res)




if __name__ == "__main__":

    logger = VisdomLogger("data", env=JOB)
    train_dataset, val_dataset, train_step, val_step = load_train_val(
        [tasks.rgb, tasks.normal, tasks.principal_curvature, tasks.rgb(size=512)],
        batch_size=32,
    )
    print ("created dataset")
    logger.add_hook(lambda logger, data: logger.step(), freq=32)

    for i, _ in enumerate(train_dataset):
        logger.update("epoch", i)
