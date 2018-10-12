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
    building_tags = np.genfromtxt(open(csv_file), delimiter=",", dtype=str, skip_header=True)
    
    test_buildings = ["almena", "mifflintown"]
    buildings = [file[6:-7] for file in glob.glob("/data/*_normal")]
    train_buildings, val_buildings = train_test_split(buildings, test_size=0.1)
    

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


def plot_images(model, logger, test_set, ood_images=None, mask_val=0.502, loss_model=None):

    preds, targets, losses, _ = model.predict_with_data(test_set)
    test_masks = build_mask(targets, mask_val, tol=1e-6)
    logger.images(test_masks.float(), "masks", resize=128)
    print ("limits R: ", preds[:, 0].min().cpu().data.numpy().mean(), preds[:, 0].mean().cpu().data.numpy().mean(), preds[:, 0].max().cpu().data.numpy().mean())
    print ("limits G: ", preds[:, 1].min().cpu().data.numpy().mean(), preds[:, 1].mean().cpu().data.numpy().mean(), preds[:, 1].max().cpu().data.numpy().mean())
    print ("limits B: ", preds[:, 2].min().cpu().data.numpy().mean(), preds[:, 2].mean().cpu().data.numpy().mean(), preds[:, 2].max().cpu().data.numpy().mean())

    print ("targets R: ", targets[:, 0].min().cpu().data.numpy().mean(), targets[:, 0].mean().cpu().data.numpy().mean(), targets[:, 0].max().cpu().data.numpy().mean())
    print ("targets G: ", targets[:, 1].min().cpu().data.numpy().mean(), targets[:, 1].mean().cpu().data.numpy().mean(), targets[:, 1].max().cpu().data.numpy().mean())
    print ("targets B: ", targets[:, 2].min().cpu().data.numpy().mean(), targets[:, 2].mean().cpu().data.numpy().mean(), targets[:, 2].max().cpu().data.numpy().mean())

    logger.images(preds.clamp(min=0, max=1), "predictions", nrow=1, resize=512)
    logger.images(targets, "targets", nrow=1, resize=512)

    if ood_images is not None:
        ood_preds = model.predict(ood_images)
        logger.images(ood_preds, "ood_predictions", nrow=1, resize=512)

    if loss_model is not None:
        with torch.no_grad():
            curvature_preds = loss_model(preds)
            curvature_targets = loss_model(targets)
            logger.images(curvature_preds.clamp(min=0, max=1), "loss_predictions", resize=128)
            logger.images(curvature_targets.clamp(min=0, max=1), "loss_targets", resize=128)

