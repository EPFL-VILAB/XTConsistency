import numpy as np
import random, sys, os, time, glob, math, itertools
from collections import defaultdict
from sklearn.model_selection import train_test_split
import parse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.checkpoint import checkpoint as util_checkpoint

from utils import *
from modules.unet import UNet, UNetOld2
from modules.percep_nets import Dense1by1Net
from modules.depth_nets import UNetDepth

from PIL import Image
import json
from scipy import ndimage

model_types = {
    ('normal', 'principal_curvature'): lambda : Dense1by1Net(),
    ('normal', 'depth_zbuffer'): lambda : UNetDepth(),
    ('normal', 'reshading'): lambda : UNet(downsample=4),
    ('depth_zbuffer', 'normal'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('reshading', 'normal'): lambda : UNet(downsample=4, in_channels=3, out_channels=3),
    ('sobel_edges', 'principal_curvature'): lambda : UNet(downsample=5, in_channels=1, out_channels=3),
    ('depth_zbuffer', 'principal_curvature'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('principal_curvature', 'depth_zbuffer'): lambda : UNet(downsample=6, in_channels=3, out_channels=1),
}

edges = {
    ('normal', 'principal_curvature'): 
        (lambda: Dense1by1Net(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth"),
    ('normal', 'depth_zbuffer'): 
        (lambda: UNetDepth(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth"),
    ('principal_curvature', 'normal'): 
        (lambda: UNetOld2(), f"{MODELS_DIR}/results_inverse_cycle_unet1x1model.pth"),
    ('depth_zbuffer', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=4), f"{MODELS_DIR}/depth2normal_unet4.pth"),
    ('principal_curvature', 'sobel_edges'): 
        (lambda: UNet(downsample=4, out_channels=1), f"{MODELS_DIR}/principal_curvature2sobel_edges.pth"),
    ('sobel_edges', 'principal_curvature'): 
        (lambda: UNet(downsample=4, in_channels=1), f"{MODELS_DIR}/sobel_edges2principal_curvature.pth"),
}

# Task output space
class Task:
    def __init__(self, name, shape=(3, 256, 256), mask_val=-1, file_name=None,
                    transform=None, file_loader=None, is_image=True, loss_func=None):
        self.name = name
        self.shape = shape
        self.mask_val = mask_val
        if transform is None: transform = (lambda x: x)
        self.transform = transform
        if file_loader is None:
            self.file_loader = lambda path: Image.open(path)
        self.file_name = file_name
        if file_name == None: self.file_name = lambda : self.name
        def mse_loss(pred, target):
            mask = build_mask(target.detach(), val=self.mask_val)
            mse = F.mse_loss(pred*mask.float(), target*mask.float())
            return mse, (mse.detach(),)

        self.loss_func = loss_func
        if loss_func is None:
            self.loss_func = mse_loss 
        self.is_image = is_image
        # output_shape, mask_val, transform

class Transfer:
    def __init__(self, src_task, dest_task, checkpoint=True, name=""):
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task = TASK_MAP[src_task]
            dest_task = TASK_MAP[dest_task]

        self.src_task = src_task
        self.dest_task = dest_task
        self.name = name
        if name == "":
            self.name = f"{src_task.name}2{dest_task.name}"
        model_type, path = edges[(src_task.name, dest_task.name)]
        self.model = DataParallelModel.load(model_type().cuda(), path)
    
    def __call__(self, argv, prefix="", plot_dict=None):
        x = argv
        num_args = 1
        if isinstance(argv, tuple):
            num_args = len(argv)
            if num_args >= 1: x = argv[0]
            if num_args >= 2: prefix = argv[1]
            if num_args >= 3: plot_dict = argv[2]

        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        total_name = f'{self.name}({prefix})'     
        if plot_dict is not None:
            plot_dict[total_name] = preds
        res = [x]
        if num_args >= 2: res.append(total_name)
        if num_args >= 3: res.append(plot_dict)
        return tuple(res)

def get_model(src_task, dest_task):
    
    if isinstance(src_task, str) and isinstance(dest_task, str):
        src_task = TASK_MAP[src_task]
        dest_task = TASK_MAP[dest_task]

    if (src_task.name, dest_task.name) in model_types:
        return model_types[(src_task.name, dest_task.name)]()
    if src_task.is_image and dest_task.is_image:
        return UNet(downsample=5, in_channels=src_task.shape[0], out_channels=dest_task.shape[0])
    # if src_task.is_image and dest_task.name == 'class_scene':

    # # if src_task.is_image and not dest_task.is_image:
    # #     return 
    return None

def load_points(path):
    with open(path) as f:
        data = json.load(f)
        points = data['vanishing_points_gaussian_sphere']
        res = []
        for a in ['x', 'y', 'z']:
            res.extend(points[a])
        return np.array(res)

def load_classes(path):
    return np.load(path)

def load_edges(path):
    Image.open(convert_path(path, 'rgb'))

def zdepth_transform(x):
    x = x.unsqueeze(0).float()
    mask = build_mask(x, 65535.0, tol=1000)
    x[~mask] = 8000.0
    x = x/8000.0
    return x[0].clamp(min=0, max=1)

def keypoints_transform(x):
    x = x.unsqueeze(0).float()
    x = x / 64131.0
    return x[0].clamp(min=0, max=1)

def semantic_loss(pred, target):
    bce = F.cross_entropy(pred, target.squeeze(1).long())
    return bce, (bce.detach(),)

def class_loss(pred, target):
    loss = F.nll_loss(pred, target)
    return loss, (loss.detach(),)

def sobel_transform(x):
    image = x.data.cpu().numpy().mean(axis=0)
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    sx = ndimage.sobel(blur, axis=0, mode='constant')
    sy = ndimage.sobel(blur, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    edge = torch.FloatTensor(sob).unsqueeze(0)
    return edge

def create_tasks():
    tasks = [
        Task('rgb'),
        Task('normal', 
            mask_val=0.502),
        Task('principal_curvature', 
            mask_val=0.0),
        Task('depth_zbuffer', 
            shape=(1, 256, 256), 
            mask_val=1.0, 
            transform=zdepth_transform),
        Task('reshading', 
            mask_val=0.0507),
        Task('edge_occlussion',
            shape=(1, 256, 256),
            ),
        Task('edge_texture',
            shape=(1, 256, 256)),
        Task('sobel_edges',
            shape=(1, 256, 256),
            transform=sobel_transform,
            file_name=(lambda : 'rgb'),
            ),
        Task('segment_semantic',
            shape=(16, 256, 256),
            loss_func=semantic_loss,
            ),
        Task('keypoints3d',
            shape=(1, 256, 256),
            transform=keypoints_transform
            ),
        Task('class_scene',
            shape=(356,),
            file_loader=(lambda x: np.load(x)),
            is_image=False,

            ),
        Task('point_info',
            shape=(9,),
            file_loader=load_points,
            is_image=False
            ),
        ]
    task_map = {}
    for task in tasks:
        task_map[task.name] = task
    return task_map

TASK_MAP = create_tasks()