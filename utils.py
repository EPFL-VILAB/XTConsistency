#utils.py

import numpy as np
import random, sys, os, time, glob, math
import random

EXPERIMENT, RESUME_JOB = open("scripts/jobinfo.txt").read().strip().split(', ')
JOB = "_".join(EXPERIMENT.split("_")[0:-1])

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	from torch.autograd import Variable

	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except:
	pass

def batch(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield arr
			arr = []
	if len(arr) != 0:
		yield arr

def batched(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield list(zip(*arr))
			arr = []
	if len(arr) != 0:
		yield list(zip(*arr))

def elapsed(times=[time.time()]):
	times.append(time.time())
	return times[-1] - times[-2]

# Cycles through iterable without making extra copies
def cycle(iterable):
	while True:
		for i in iterable:
			yield i

def build_mask(target, val=0.0, tol=1e-3):
	if target.shape[1] == 1:
		return ~((target >= val - tol) & (target <= val + tol))
	
	mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
	mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
	mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
	mask = ~(mask1 & mask2 & mask3).unsqueeze(1).expand_as(target)
	return mask


def load_data(csv_file, source_task, dest_task, batch_size=32):

    building_tags = np.genfromtxt(open("data/train_val_test_fullplus.csv"), delimiter=",", dtype=str, skip_header=True)
    test_buildings = ["almena", "mifflintown"]
    train_buildings = [building for building, train, test, val in building_tags \
                            if train == "1" and building not in test_buildings]
    val_buildings = [building for building, train, test, val in building_tags if val == "1"]
    

    train_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=train_buildings, source_task=source_task, dest_task=dest_task),
        batch_size=batch_size,
        num_workers=16,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=val_buildings, source_task=source_task, dest_task=dest_task),
        batch_size=batch_size,
        num_workers=16,
        shuffle=True,
    )
    test_loader1 = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=["almena"], source_task=source_task, dest_task=dest_task),
        batch_size=6,
        num_workers=12,
        shuffle=False,
    )
    test_loader2 = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=["mifflintown"], source_task=source_task, dest_task=dest_task),
        batch_size=6,
        num_workers=6,
        shuffle=False,
    )
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir="data/ood_images"),
        batch_size=10,
        num_workers=10,
        shuffle=False,
    )
    train_loader, val_loader = cycle(train_loader), cycle(val_loader)
    test_set = list(itertools.islice(test_loader1, 1)) + list(itertools.islice(test_loader2, 1))
    test_images = torch.cat([x for x, y in test_set], dim=0)
    ood_images = list(itertools.islice(ood_loader, 1))

    return train_loader, val_loader, test_set, test_images, ood_images
