
import numpy as np
import random, sys, os, time, glob, math, itertools, yaml
from sklearn.model_selection import train_test_split
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
dtype = torch.cuda.FloatTensor


EXPERIMENT, RESUME_JOB, BASE_DIR = open("scripts/jobinfo.txt").read().strip().split(', ')
JOB = "_".join(EXPERIMENT.split("_")[0:-1])

MODELS_DIR = f"{BASE_DIR}/shared/models"
DATA_DIRS = [f"{BASE_DIR}/data/taskonomy3", f"{BASE_DIR}/small_data"]
RESULTS_DIR = f"{BASE_DIR}/shared/results_{EXPERIMENT}"
SHARED_DIR = f"{BASE_DIR}/shared"

if BASE_DIR == "/":
    DATA_DIRS = ["/data", "/edge_1", "/edges_1", "/edges_2", "/edges_3", "/reshade", "/semantic5", "/keypoints", "/keypoints2d", "/class"]
    RESULTS_DIR = "/result"
    MODELS_DIR = "/models"
else:
    os.system(f"sudo mkdir -p {RESULTS_DIR}")


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

# def random_resize(iterable, vals=[128, 192, 256, 320]):
#    """ Cycles through iterable while randomly resizing batch values. """
#     from transforms import resize
#     while True:
#         for X, Y in iterable:
#             val = random.choice(vals)
#             yield resize(X.to(DEVICE), val=val).detach(), resize(Y.to(DEVICE), val=val).detach()


def get_files(exp, data_dirs=DATA_DIRS):
    """ Gets data files across mounted directories matching glob expression pattern. """

    files, seen = [], set()
    for data_dir in data_dirs:
        for file in glob.glob(f'{data_dir}/{exp}'):
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
    def dest_transforms(image):
        image = image.data.cpu().numpy().mean(axis=0)
        blur = ndimage.filters.gaussian_filter(image, sigma=2)
        sx = ndimage.sobel(blur, axis=0, mode='constant')
        sy = ndimage.sobel(blur, axis=1, mode='constant')
        image = np.hypot(sx, sy)
        edge = torch.FloatTensor(image, device=x.device).unsqueeze(0)
        return edge

    edges = torch.stack([dest_transforms(image) for image in x], dim=0).to(x.device)
    return edges
