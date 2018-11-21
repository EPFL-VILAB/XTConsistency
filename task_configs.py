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
from torchvision import transforms

from utils import *
from models import DataParallelModel
from modules.unet import UNet, UNetOld2, UNetOld
from modules.percep_nets import Dense1by1Net
from modules.depth_nets import UNetDepth

from PIL import Image
import json
from scipy import ndimage


model_types = {
    ('normal', 'principal_curvature'): lambda : Dense1by1Net(),
    ('normal', 'depth_zbuffer'): lambda : UNetDepth(),
    ('normal', 'reshading'): lambda : UNet(downsample=5),
    ('depth_zbuffer', 'normal'): lambda : UNet(downsample=6, in_channels=1, out_channels=3),
    ('reshading', 'normal'): lambda : UNet(downsample=4, in_channels=3, out_channels=3),
    ('sobel_edges', 'principal_curvature'): lambda : UNet(downsample=5, in_channels=1, out_channels=3),
    ('depth_zbuffer', 'principal_curvature'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('principal_curvature', 'depth_zbuffer'): lambda : UNet(downsample=6, in_channels=3, out_channels=1),
}
# Task output space
class Task:
    def __init__(self, name, shape=(3, 256, 256), mask_val=-1, file_name=None, plot_func=None,
                    transform=None, file_loader=None, is_image=True, loss_func=None):
        self.name = name
        self.shape = shape
        self.mask_val = mask_val
        if transform is None: transform = (lambda x: x)
        self.transform = transform
        self.file_loader = file_loader
        if file_loader is None:
            self.file_loader = lambda path: Image.open(path)
        self.file_name = file_name
        if file_name == None: self.file_name = lambda : self.name
        def mse_loss(pred, target):
            mask = build_mask(target.detach(), val=self.mask_val)
            mse = F.mse_loss(pred*mask.float(), target*mask.float())
            return mse, (mse.detach(),)

        self.plot_func = plot_func
        if plot_func == None:
            self.plot_func = lambda x, name, logger: logger.images(x.clamp(min=0, max=1), name, nrow=2, resize=256)


        self.loss_func = loss_func
        if loss_func is None:
            self.loss_func = mse_loss 
        self.is_image = is_image
        # output_shape, mask_val, transform
    def __eq__(self, other):
        return self.name == other.name
        
def get_model(src_task, dest_task, easy=False):
    
    if isinstance(src_task, str) and isinstance(dest_task, str):
        src_task = TASK_MAP[src_task]
        dest_task = TASK_MAP[dest_task]

    downsample = 4 if easy else 5
    if (src_task.name, dest_task.name) in model_types:
        return model_types[(src_task.name, dest_task.name)]()
    if src_task.is_image and dest_task.is_image:
        return UNet(downsample=downsample, in_channels=src_task.shape[0], out_channels=dest_task.shape[0])
    
    # if src_task.is_image and dest_task.name == 'class_scene':

    # if src_task.is_image and not dest_task.is_image:
    #     return 
    
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

def load_curv_norm(path):
    curv = transforms.ToTensor()(transforms.Resize(256, interpolation=PIL.Image.NEAREST)(Image.open(convert_path(path, 'principal_curvature'))))
    norm = transforms.ToTensor()(transforms.Resize(256, interpolation=PIL.Image.NEAREST)(Image.open(convert_path(path, 'normal'))))
    res =  (torch.cat([curv, norm]))
    # print(res.shape)
    return res

def plot_curv_norm(x, name, logger):
    print(x.shape)
    curv, norm = x[:,:3,:,:], x[:,3:,:,:]
    logger.images(curv.clamp(min=0, max=1), f'{name}_curv', nrow=2, resize=256)
    logger.images(norm.clamp(min=0, max=1), f'{name}_norm', nrow=2, resize=256)

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

def keypoints2d_transform(x):
    x = x.unsqueeze(0).float()
    x = x / 2400.0
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
        Task('edge_occlusion',
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
        Task('keypoints2d',
            shape=(1, 256, 256),
            transform=keypoints2d_transform
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
        Task('imagenet_percep',
            shape=(2048,),
            file_loader=None,
            is_image=False
            ),
        Task('random_network', 
            mask_val=-1.0),
        Task('curv_norm',
            file_loader=load_curv_norm,
            shape=(6, 256, 256),
            plot_func=plot_curv_norm,
            file_name=(lambda: 'principal_curvature')
            ),
        ]
    task_map = {}
    for task in tasks:
        task_map[task.name] = task
    return task_map

TASK_MAP = create_tasks()