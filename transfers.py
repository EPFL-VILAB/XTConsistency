
import os, sys, math, random, itertools
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as util_checkpoint
from torchvision import models

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, task_map, get_model, Task, RealityTask

from modules.resnet import ResNet
from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet, UNetOld, UNetOld2, UNetReshade

from fire import Fire
import IPython


pretrained_transfers = {
    ('normal', 'principal_curvature'): 
        (lambda: Dense1by1Net(), f"{MODELS_DIR}/normal2curvature_dense_1x1.pth"),
    ('normal', 'depth_zbuffer'): 
        (lambda: UNetDepth(), f"{MODELS_DIR}/normal2zdepth_unet_v4.pth"),
    
    ('normal', 'sobel_edges'): 
        (lambda: UNet(out_channels=1, downsample=4).cuda(), f"{MODELS_DIR}/normal2edges2d_sobel_unet4.pth"),
    ('sobel_edges', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=5).cuda(), f"{MODELS_DIR}/sobel_edges2normal.pth"),

    ('normal', 'grayscale'): 
        (lambda: UNet(out_channels=1, downsample=6).cuda(), f"{MODELS_DIR}/normals2gray_unet.pth"),
    ('principal_curvature', 'normal'): 
        (lambda: UNetOld2(), f"{MODELS_DIR}/results_inverse_cycle_unet1x1model.pth"),
    ('principal_curvature', 'sobel_edges'): 
        (lambda: UNet(downsample=4, out_channels=1), f"{MODELS_DIR}/principal_curvature2sobel_edges.pth"),
    # ('depth_zbuffer', 'normal'): 
    #     (lambda: UNet(in_channels=1, downsample=4), f"{MODELS_DIR}/depth2normal_unet4.pth"),
    ('depth_zbuffer', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=6), f"{MODELS_DIR}/depth2normal_unet6.pth"),
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
        # (lambda: UNetOld(), f"{MODELS_DIR}/unet_baseline.pth"),
        (lambda: UNet(), f"{MODELS_DIR}/unet_baseline_standardval.pth"),
    ('rgb', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{MODELS_DIR}/rgb2principal_curvature.pth"),
    ('rgb', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/rgb2keypoints2d.pth"),
    ('rgb', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{MODELS_DIR}/rgb2reshade.pth"),
    ('rgb', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, out_channels=1), f"{MODELS_DIR}/rgb2zdepth_buffer.pth"),

    ('keypoints2d', 'principal_curvature'):
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/keypoints2d2principal_curvature_temp.pth"),
    

    ('keypoints3d', 'principal_curvature'):
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/keypoints3d2principal_curvature.pth"),
    ('principal_curvature', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/principal_curvature2keypoints3d.pth"),

    # ('normal', 'reshading'):
    #     (lambda: UNetReshade(downsample=4), f"{MODELS_DIR}/normal2reshading_unet4.pth"),
    ('normal', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{MODELS_DIR}/normal2reshade_unet5.pth"),
    ('reshading', 'normal'):
        (lambda: UNet(downsample=4), f"{MODELS_DIR}/reshading2normal.pth"),

    ('sobel_edges', 'reshading'):
        (lambda: UNetReshade(downsample=5, in_channels=1), f"{MODELS_DIR}/sobel_edges2reshading.pth"),

    ('normal', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/normal2keypoints3d.pth"),
    ('keypoints3d', 'normal'):
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/keypoints3d2normal.pth"),

    ('normal', 'imagenet_percep'):
        (lambda: ResNetClass(), None),
    ('normal', 'random_network'):
        (lambda: UNet(downsample=4), None),
}

class Transfer(object):
    
    def __init__(self, src_task, dest_task, checkpoint=True, name=None, model_type=None, path=None):
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task, dest_task = get_task(src_task), get_task(dest_task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        saved_type, saved_path = None, None
        if model_type is None and path is None:
            saved_type, saved_path = pretrained_transfers.get((src_task.name, dest_task.name), (None, None))

        self.model_type, self.path = model_type or saved_type, path or saved_path

        if self.model_type is None:
            path = f"{MODELS_DIR}/{src_task.name}2{dest_task.name}.pth"
            if os.path.exists(path):
                self.model_type, self.path = lambda: get_model(src_task, dest_task), path

        self.model = None
    
    def load_model(self, optimizer=True):
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.model_type().to(DEVICE), self.path)
                if optimizer:
                    self.model.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
            else:
                self.model = self.model_type()
        return self.model

    def __call__(self, x):
        self.load_model()
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        preds.task = self.dest_task
        return preds

    def __repr__(self):
        return self.name or str(self.src_task) + " -> " + str(self.dest_task)


class RealityTransfer(Transfer):

    def __init__(self, src_task, dest_task):
        super().__init__(src_task, dest_task, model_type=lambda: None)

    def load_model(self, optimizer=True):
        pass

    def __call__(self, x):
        assert (isinstance(self.src_task, RealityTask))
        return self.src_task.task_data[self.dest_task].to(DEVICE)


class FineTunedTransfer(Transfer):
    
    def __init__(self, transfer):
        super().__init__(transfer.src_task, transfer.dest_task, checkpoint=transfer.checkpoint, name=transfer.name)
        self.cached_models = {}

    def load_model(self, parents=[]):

        model_path = get_finetuned_model_path(parents + [self])

        if model_path not in self.cached_models: 
            if not os.path.exists(model_path):
                print(f"{model_path} not found, loading pretrained")
                self.cached_models[model_path] = super().load_model()
            else:
                print(f"{model_path} found, loading finetuned")
                self.cached_models[model_path] = DataParallelModel.load(self.model_type().cuda(), model_path)
        
        self.model = self.cached_models[model_path]
        return self.model

    def __call__(self, x):

        if not hasattr(x, "parents"): 
            x.parents = []

        self.load_model(parents=x.parents)
        preds = util_checkpoint(self.model, x) if self.checkpoint else self.model(x)
        preds.parents = x.parents + ([self])
        return preds



functional_transfers = (
    Transfer('normal', 'principal_curvature', name='f'),
    Transfer('principal_curvature', 'normal', name='F'),

    Transfer('normal', 'depth_zbuffer', name='g'),
    Transfer('depth_zbuffer', 'normal', name='G'),

    Transfer('normal', 'sobel_edges', name='s'),
    Transfer('sobel_edges', 'normal', name='S'),
    
    Transfer('principal_curvature', 'sobel_edges', name='CE'),
    Transfer('sobel_edges', 'principal_curvature', name='EC'),

    Transfer('depth_zbuffer', 'sobel_edges', name='DE'),
    Transfer('sobel_edges', 'depth_zbuffer', name='ED'),

    Transfer('principal_curvature', 'depth_zbuffer', name='h'),
    Transfer('depth_zbuffer', 'principal_curvature', name='H'),

    Transfer('rgb', 'normal', name='n'),
    Transfer('rgb', 'normal', name='npstep', 
        model_type=lambda: UNetOld(),
        path=f"{MODELS_DIR}/unet_percepstep_0.1.pth",
    ),
    Transfer('rgb', 'principal_curvature', name='RC'),
    Transfer('rgb', 'keypoints2d', name='k'),
    Transfer('rgb', 'sobel_edges', name='a'),
    Transfer('rgb', 'reshading', name='r'),
    Transfer('rgb', 'depth_zbuffer', name='d'),

    Transfer('keypoints2d', 'principal_curvature', name='KC'),

    Transfer('keypoints3d', 'principal_curvature', name='k3C'),
    Transfer('principal_curvature', 'keypoints3d', name='Ck3'),

    Transfer('normal', 'reshading', name='nr'),
    Transfer('reshading', 'normal', name='rn'),
 
    Transfer('keypoints3d', 'normal', name='k3N'),
    Transfer('normal', 'keypoints3d', name='Nk3'),

    Transfer('sobel_edges', 'reshading', name='Er'),
    # Transfer('normal', 'imagenet_percep', name='NIm'),
    # Transfer('normal', 'random_network', name='RND'),
)

finetuned_transfers = [FineTunedTransfer(transfer) for transfer in functional_transfers]
TRANSFER_MAP = {t.name:t for t in functional_transfers}
functional_transfers = namedtuple('functional_transfers', TRANSFER_MAP.keys())(**TRANSFER_MAP)

def get_named_transfer(transfer):
    for t in functional_transfers:
        if transfer.src_task == t.src_task and transfer.dest_task == t.dest_task:
            return t
    return transfer

(f, F, g, G, s, S, CE, EC, DE, ED, h, H, n, npstep, RC, k, a, r, d, KC, k3C, Ck3, nr, rn, k3N, Nk3, Er) = functional_transfers

if __name__ == "__main__":
    y = g(F(f(x)))
    print (y.shape)
    





