
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as util_checkpoint

from utils import *
from models import TrainableModel, DataParallelModel

from modules.resnet import ResNet
from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet, UNetOld, UNetOld2
from sklearn.model_selection import train_test_split
from fire import Fire

from skimage import feature
from functools import partial
from task_configs import TASK_MAP

import IPython


pretrained_transfers = {
    ('normal', 'principal_curvature'): 
        (lambda: Dense1by1Net(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth"),
    ('normal', 'depth_zbuffer'): 
        (lambda: UNetDepth(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth"),
    ('normal', 'sobel_edges'): 
        (lambda: UNet(out_channels=1, downsample=4).cuda(), f"{MODELS_DIR}/normal2edges2d_sobel_unet4.pth"),
    ('normal', 'grayscale'): 
        (lambda: UNet(out_channels=1, downsample=6).cuda(), f"{MODELS_DIR}/normals2gray_unet.pth"),
    ('principal_curvature', 'normal'): 
        (lambda: UNetOld2(), f"{MODELS_DIR}/results_inverse_cycle_unet1x1model.pth"),
    ('principal_curvature', 'sobel_edges'): 
        (lambda: UNet(downsample=4, out_channels=1), f"{MODELS_DIR}/principal_curvature2sobel_edges.pth"),
    ('depth_zbuffer', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=4), f"{MODELS_DIR}/depth2normal_unet4.pth"),
    ('depth_zbuffer', 'sobel_edges'): 
        (lambda: UNet(downsample=4, in_channels=1, out_channels=1).cuda(), f"{MODELS_DIR}/depth_zbuffer2sobel_edges.pth"),
    ('sobel_edges', 'principal_curvature'): 
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/sobel_edges2principal_curvature.pth"),
    ('rgb', 'sobel_edges'):
        (lambda: sobel_kernel, None),
    ('sobel_edges', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=1), f"{MODELS_DIR}/sobel_edges2depth_zbuffer.pth"),
    ('principal_curvature', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, out_channels=1), f"{MODELS_DIR}/principal_curvature2depth_zbuffer.pth"),
    ('depth_zbuffer', 'principal_curvature'):
        (lambda: UNet(downsample=4, in_channels=1), f"{MODELS_DIR}/depth_zbuffer2principal_curvature.pth"),
    ('rgb', 'normal'):
        (lambda: UNetOld(), f"{MODELS_DIR}/mixing_percepcurv_norm.pth"),
    ('rgb', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{BASE_DIR}/shared/results_transfer_rgb2curv_3/rgb2principal_curvature.pth"),
    ('rgb', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/rgb2keypoints2d.pth"),
    ('keypoints2d', 'principal_curvature'):
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/keypoints2d2principal_curvature_temp.pth")
}

class Transfer(object):
    
    def __init__(self, src_task, dest_task, checkpoint=True, name=None):
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task = TASK_MAP[src_task]
            dest_task = TASK_MAP[dest_task]

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}" 
        self.model_type, self.path = pretrained_transfers[(src_task.name, dest_task.name)]
        self.model = None
    
    def __call__(self, x):
        
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.model_type().cuda(), self.path)
            else:
                self.model = self.model_type()

        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        return preds


functional_transfers = (
    Transfer('normal', 'principal_curvature', name='f'),
    Transfer('principal_curvature', 'normal', name='F'),
    Transfer('normal', 'depth_zbuffer', name='g'),
    Transfer('depth_zbuffer', 'normal', name='G'),
    Transfer('normal', 'sobel_edges', name='s'),
    Transfer('principal_curvature', 'sobel_edges', name='CE'),
    Transfer('sobel_edges', 'principal_curvature', name='EC'),
    Transfer('depth_zbuffer', 'sobel_edges', name='DE'),
    Transfer('rgb', 'sobel_edges', name='a'),
    Transfer('sobel_edges', 'depth_zbuffer', name='ED'),
    Transfer('principal_curvature', 'depth_zbuffer', name='h'),
    Transfer('depth_zbuffer', 'principal_curvature', name='H'),
    Transfer('rgb', 'normal', name='n'),
    Transfer('rgb', 'keypoints2d', name='k'),
    Transfer('keypoints2d', 'principal_curvature', name='KC'),
    Transfer('rgb', 'principal_curvature', name='RC'),
)

# (f, F, g, G, s, CE, EC, DE, a, ED, h, H, n) = functional_transfers



"""
curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
curvature_model = api(curvature_model_base)

curvature2normal_base = DataParallelModel.load(UNetOld2().cuda(), f"{MODELS_DIR}/results_inverse_cycle_unet1x1model.pth")
curvature2normal = api(curvature2normal_base)

curve2depth_base = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/alpha_train_triangle_curve2depth.pth")
curve2depth = api(curve2depth_base)

depth_model_base = DataParallelModel.load(UNetDepth().cuda(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth")
depth_model = api(depth_model_base)

depth2curve_base = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/alpha_train_triangle_depth2curve.pth")
depth2curve = api(depth2curve_base)

depth2normal_base = DataParallelModel.load(UNet(in_channels=1, downsample=4).cuda(), f"{MODELS_DIR}/depth2normal_unet4.pth")
depth2normal = api(depth2normal_base)

# normal2reshade_base = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/normal2reshade_unet.pth")
# normal2reshade = api(normal2reshade_base)

# reshade2normal_base = DataParallelModel.load(UNet(downsample=5).cuda(), f"{MODELS_DIR}/reshading2normal_unet5.pth")
# reshade2normal = api(normal2reshade_base)

curvature2edges_base = DataParallelModel.load(UNet(downsample=4, out_channels=1).cuda(), f"{MODELS_DIR}/principal_curvature2sobel_edges.pth")
curvature2edges = api(curvature2edges_base)

depth2edges_base = DataParallelModel.load(UNet(downsample=4, in_channels=1, out_channels=1).cuda(), f"{MODELS_DIR}/depth_zbuffer2sobel_edges.pth")
depth2edges = api(depth2edges_base)

edges2curvature_base = DataParallelModel.load(UNet(downsample=5, in_channels=1, out_channels=3).cuda(), f"{MODELS_DIR}/sobel_edges2principal_curvature.pth")
edges2curvature = api(edges2curvature_base)

def curve_cycle(pred, checkpoint=True):
    return depth2curve(depth_model(pred, checkpoint=checkpoint).expand(-1, 3, -1, -1), checkpoint=checkpoint)

def depth_cycle(pred, checkpoint=True):
    return curve2depth(curvature_model(pred, checkpoint=checkpoint), checkpoint=checkpoint).mean(dim=1, keepdim=True)

# normal2edge_base = DataParallelModel.load(UNetOld(out_channels=1).cuda(), f"{MODELS_DIR}/normal2fakeedges.pth")
# def normal2edge(pred):
#     return checkpoint(normal2edge_base, pred)

normal2edge_base = DataParallelModel.load()
normal2edge = api(normal2edge_base)

# normal2rgb_base = DataParallelModel.load(UNet(downsample=6).cuda(), "mount/shared/results_normals2rgb_unet_5/model.pth")
# def normal2rgb(pred):
#     return checkpoint(normal2rgb_base, pred)

normal2gray_base = DataParallelModel.load(UNet(out_channels=1, downsample=6).cuda(), f"{MODELS_DIR}/normals2gray_unet.pth")
normal2gray = api(normal2gray_base)
"""

