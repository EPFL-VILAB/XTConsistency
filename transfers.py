
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from utils import *
from models import TrainableModel, DataParallelModel

from modules.resnet import ResNet
from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet, UNetOld
from sklearn.model_selection import train_test_split
from fire import Fire

from skimage import feature
from functools import partial

import IPython

curvature_model_base = DataParallelModel.load(Dense1by1Net().cuda(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth")
def curvature_model(pred):
    return checkpoint(curvature_model_base, pred)

# curvature2normal_base = DataParallelModel.load(UNet().cuda(), f"{MODELS_DIR}/results_inverse_cycle_unet1x1model.pth")
# def curvature2normal(pred):
#     return checkpoint(curvature2normal_base, pred)

curve2depth_base = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/alpha_train_triangle_curve2depth.pth")
def curve2depth(pred):
    return checkpoint(curve2depth_base, pred)

depth_model_base = DataParallelModel.load(UNetDepth().cuda(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth")
def depth_model(pred):
    return checkpoint(depth_model_base, pred)

depth2curve_base = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/alpha_train_triangle_depth2curve.pth")
def depth2curve(pred):
    return checkpoint(depth2curve_base, pred)

def curve_cycle(pred):
    return depth2curve(depth_model(pred).expand(-1, 3, -1, -1))

def depth_cycle(pred):
    return curve2depth(curvature_model(pred)).mean(dim=1, keepdim=True)

# normal2edge_base = DataParallelModel.load(UNetOld(out_channels=1).cuda(), f"{MODELS_DIR}/normal2fakeedges.pth")
# def normal2edge(pred):
#     return checkpoint(normal2edge_base, pred)

normal2edge_base = DataParallelModel.load(UNet(out_channels=1, downsample=4).cuda(), f"{MODELS_DIR}/normal2edges2d_sobel_unet4.pth")
def normal2edge_detail(pred):
    return checkpoint(normal2edge_base, pred)

normal2rgb_base = DataParallelModel.load(UNet(downsample=6).cuda(), "mount/shared/results_normals2rgb_unet_5/model.pth")
def normal2rgb(pred):
    return checkpoint(normal2rgb_base, pred)

normal2gray_base = DataParallelModel.load(UNet(out_channels=1, downsample=6).cuda(), "mount/shared/results_normals2gray_unet_5/model.pth")
def normal2gray(pred):
    return checkpoint(normal2gray_base, pred)


