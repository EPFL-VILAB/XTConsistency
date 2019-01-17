
import numpy as np
import random, sys, os, time, glob, math, itertools, yaml
import parse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from functools import partial
from scipy import ndimage

import IPython

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

EXPERIMENT, RESUME_JOB, BASE_DIR = open("scripts/jobinfo.txt").read().strip().split(', ')
JOB = "_".join(EXPERIMENT.split("_")[0:-1])

MODELS_DIR = f"{BASE_DIR}/shared/models"
DATA_DIRS = [f"{BASE_DIR}/data/taskonomy3", f"{BASE_DIR}/small_data"]
RESULTS_DIR = f"{BASE_DIR}/shared/results_{EXPERIMENT}"
SHARED_DIR = f"{BASE_DIR}/shared"
OOD_DIR = f"{SHARED_DIR}/ood_standard_set"
USE_RAID = False

if BASE_DIR == "/":
    DATA_DIRS = ["/data", "/edge_1", "/edges_1", "/edges_2", "/edges_3", "/reshade", "/semantic5", "/keypoints", "/keypoints2d", "/class"]
    RESULTS_DIR = "/result"
    MODELS_DIR = "/models"
elif BASE_DIR == "locals":
    DATA_DIRS = ["local/small_data"]
    RESULTS_DIR = "local/result"
    MODELS_DIR = "local/models"
elif BASE_DIR == "cvgl":
    DATA_DIRS = ["/cvgl/group/taskonomy/processed"]
    RESULTS_DIR = "local/result"
    MODELS_DIR = "local/models"
elif BASE_DIR == "raid":
    USE_RAID = True
    BASE_DIR = "/raid/scratch/rsuri2"
    DATA_DIRS = ["/raid/scratch/tstand/taskonomy/"]
    RESULTS_DIR = f"results/results_{EXPERIMENT}"
    MODELS_DIR = "/raid/scratch/rsuri2/models"
    OOD_DIR = "/cvgl/group/taskonomy/taskconsistency/ood_standard_set"
    os.system(f"mkdir -p {RESULTS_DIR}")
else:
    os.system(f"sudo mkdir -p {RESULTS_DIR}")

print (DATA_DIRS)

def elapsed(last_time=[time.time()]):
    """ Returns the time passed since elapsed() was last called. """
    current_time = time.time()
    diff = current_time - last_time[0]
    last_time[0] = current_time
    return diff

def cycle(iterable):
    """ Cycles through iterable without making extra copies. """
    while True:
        for i in iterable:
            yield i

def average(arr):
    return sum(arr) / len(arr)

# def random_resize(iterable, vals=[128, 192, 256, 320]):
#    """ Cycles through iterable while randomly resizing batch values. """
#     from transforms import resize
#     while True:
#         for X, Y in iterable:
#             val = random.choice(vals)
#             yield resize(X.to(DEVICE), val=val).detach(), resize(Y.to(DEVICE), val=val).detach()


def get_files(exp, data_dirs=DATA_DIRS, recursive=False):
    """ Gets data files across mounted directories matching glob expression pattern. """

    files, seen = [], set()
    for data_dir in data_dirs:
        for file in glob.glob(f'{data_dir}/{exp}', recursive=recursive):
            if file[len(data_dir):] not in seen:
                files.append(file)
                seen.add(file[len(data_dir):])
    return files


def get_finetuned_model_path(parents):
    if BASE_DIR == "/":
        return f"{RESULTS_DIR}/" + "_".join([parent.name for parent in parents[::-1]]) + ".pth"
    else:
        return f"{MODELS_DIR}/finetuned/" + "_".join([parent.name for parent in parents[::-1]]) + ".pth"


def plot_images(model, logger, test_set, dest_task="normal",
        ood_images=None, show_masks=False, loss_models={}, 
        preds_name=None, target_name=None, ood_name=None,
    ):

    from task_configs import get_task, ImageTask
    
    test_images, preds, targets, losses, _ = model.predict_with_data(test_set)

    if isinstance(dest_task, str):
        dest_task = get_task(dest_task)

    if show_masks and isinstance(dest_task, ImageTask):
        test_masks = ImageTask.build_mask(targets, dest_task.mask_val, tol=1e-3)
        logger.images(test_masks.float(), f"{dest_task}_masks", resize=64)

    dest_task.plot_func(preds, preds_name or f"{dest_task.name}_preds", logger)
    dest_task.plot_func(targets, target_name or f"{dest_task.name}_target", logger)
    
    if ood_images is not None:
        ood_preds = model.predict(ood_images)
        dest_task.plot_func(ood_preds, ood_name or f"{dest_task.name}_ood_preds", logger)

    for name, loss_model in loss_models.items():
        with torch.no_grad():
            output = loss_model(preds, targets, test_images)
            if hasattr(output, "task"):
                output.task.plot_func(output, name, logger, resize=128)
            else:
                logger.images(output.clamp(min=0, max=1), name, resize=128)


def gaussian_filter(channels=3, kernel_size=5, sigma=1.0, device=0):

    x_cord = torch.arange(kernel_size).float()
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance)
    )
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    return gaussian_kernel


def motion_blur_filter(kernel_size=15):
    channels = 3
    kernel_motion_blur = torch.zeros((kernel_size, kernel_size))
    kernel_motion_blur[int((kernel_size - 1) / 2), :] = torch.ones(kernel_size)
    kernel_motion_blur = kernel_motion_blur / kernel_size
    kernel_motion_blur = kernel_motion_blur.view(1, 1, kernel_size, kernel_size)
    kernel_motion_blur = kernel_motion_blur.repeat(channels, 1, 1, 1)
    return kernel_motion_blur


def sobel_kernel(x):

    blur = gaussian_filter(channels=3, kernel_size=5, sigma=2.0)
    x = F.conv2d(x, weight=blur.to(x.device), bias=None, groups=3, padding=2)
    x = x.mean(dim=1, keepdim=True)

    a=np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight=nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    conv1 = conv1.to(x.device)
    G_x=conv1(x)

    b=np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]])
    conv2=nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight=nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    conv2 = conv2.to(x.device)
    G_y=conv2(x)

    G=torch.sqrt(torch.pow(G_x,2)+ torch.pow(G_y,2))
    return G
