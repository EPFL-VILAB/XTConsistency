
import os, sys, math, random, itertools, functools
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

from modules.percep_nets import DenseNet, Dense1by1Net, DenseKernelsNet, DeepNet, BaseNet, WideNet, PyramidNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet, UNetOld, UNetOld2, UNetReshade
from modules.resnet import ResNetClass

from fire import Fire
import IPython


pretrained_transfers = {

    ('normal', 'principal_curvature'):
        (lambda: Dense1by1Net(), f"{MODELS_DIR}/normal2curvature.pth"),
    ('normal', 'depth_zbuffer'):
        (lambda: UNetDepth(), f"{MODELS_DIR}/normal2zdepth_zbuffer.pth"),
    ('normal', 'sobel_edges'):
        (lambda: UNet(out_channels=1, downsample=4).cuda(), f"{MODELS_DIR}/normal2edges2d.pth"),
    ('normal', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{MODELS_DIR}/normal2reshade.pth"),
    ('normal', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/normal2keypoints3d.pth"),
    ('normal', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/normal2keypoints2d_new.pth"),
    ('normal', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/normal2edge_occlusion.pth"),

    ('depth_zbuffer', 'normal'):
        (lambda: UNet(in_channels=1, downsample=6), f"{MODELS_DIR}/depth2normal.pth"),
    ('depth_zbuffer', 'sobel_edges'):
        (lambda: UNet(downsample=4, in_channels=1, out_channels=1).cuda(), f"{MODELS_DIR}/depth_zbuffer2sobel_edges.pth"),
    ('depth_zbuffer', 'principal_curvature'):
        (lambda: UNet(downsample=4, in_channels=1), f"{MODELS_DIR}/depth_zbuffer2principal_curvature.pth"),
    ('depth_zbuffer', 'reshading'):
        (lambda: UNetReshade(downsample=5, in_channels=1), f"{MODELS_DIR}/depth_zbuffer2reshading.pth"),
    ('depth_zbuffer', 'keypoints3d'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{MODELS_DIR}/depth_zbuffer2keypoints3d.pth"),
    ('depth_zbuffer', 'keypoints2d'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{MODELS_DIR}/depth_zbuffer2keypoints2d.pth"),
    ('depth_zbuffer', 'edge_occlusion'):
        (lambda: UNet(downsample=5, in_channels=1, out_channels=1), f"{MODELS_DIR}/depth_zbuffer2edge_occlusion.pth"),

    ('reshading', 'depth_zbuffer'):
        (lambda: UNetReshade(downsample=5, out_channels=1), f"{MODELS_DIR}/reshading2depth_zbuffer.pth"),
    ('reshading', 'keypoints2d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/reshading2keypoints2d_new.pth"),
    ('reshading', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/reshading2edge_occlusion.pth"),
    ('reshading', 'normal'):
        (lambda: UNet(downsample=4), f"{MODELS_DIR}/reshading2normal.pth"),
    ('reshading', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/reshading2keypoints3d.pth"),
    ('reshading', 'sobel_edges'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/reshading2sobel_edges.pth"),
    ('reshading', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{MODELS_DIR}/reshading2principal_curvature.pth"),

    ('rgb', 'sobel_edges'):
        (lambda: SobelKernel(), None),
    ('rgb', 'principal_curvature'):
        (lambda: UNet(downsample=5), f"{MODELS_DIR}/rgb2principal_curvature.pth"),
    ('rgb', 'keypoints2d'):
        (lambda: UNet(downsample=3, out_channels=1), f"{MODELS_DIR}/rgb2keypoints2d_new.pth"),
    ('rgb', 'keypoints3d'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/rgb2keypoints3d.pth"),
    ('rgb', 'edge_occlusion'):
        (lambda: UNet(downsample=5, out_channels=1), f"{MODELS_DIR}/rgb2edge_occlusion.pth"),
    ('rgb', 'normal'):
        (lambda: UNet(), f"{MODELS_DIR}/rgb2normal_baseline.pth"),
    ('rgb', 'reshading'):
        (lambda: UNetReshade(downsample=5), f"{MODELS_DIR}/rgb2reshading_baseline.pth"),
    ('rgb', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, out_channels=1), f"{MODELS_DIR}/rgb2zdepth_baseline.pth"),

    ('normal', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),
    ('depth_zbuffer', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),
    ('reshading', 'imagenet'):
        (lambda: ResNetClass().cuda(), None),

    ('principal_curvature', 'sobel_edges'): 
        (lambda: UNet(downsample=4, out_channels=1), f"{MODELS_DIR}/principal_curvature2sobel_edges.pth"),
    ('sobel_edges', 'depth_zbuffer'):
        (lambda: UNet(downsample=6, in_channels=1, out_channels=1), f"{MODELS_DIR}/sobel_edges2depth_zbuffer.pth"),

    ('depth_zbuffer', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=6), f"{MODELS_DIR}/depth2normal.pth"),
    ('keypoints2d', 'normal'):
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/keypoints2d2normal_new.pth"),
    ('keypoints3d', 'normal'):
        (lambda: UNet(downsample=5, in_channels=1), f"{MODELS_DIR}/keypoints3d2normal.pth"),
    ('principal_curvature', 'normal'): 
        (lambda: UNetOld2(), f"{MODELS_DIR}/principal_curvature2normal.pth"),
    ('sobel_edges', 'normal'): 
        (lambda: UNet(in_channels=1, downsample=5).cuda(), f"{MODELS_DIR}/sobel_edges2normal.pth"),
    ('edge_occlusion', 'normal'):
        (lambda: UNet(in_channels=1, downsample=5), f"{MODELS_DIR}/edge_occlusion2normal.pth"),

}

class Transfer(nn.Module):

    def __init__(self, src_task, dest_task,
        checkpoint=True, name=None, model_type=None, path=None,
        pretrained=True, finetuned=False
    ):
        super().__init__()
        if isinstance(src_task, str) and isinstance(dest_task, str):
            src_task, dest_task = get_task(src_task), get_task(dest_task)

        self.src_task, self.dest_task, self.checkpoint = src_task, dest_task, checkpoint
        self.name = name or f"{src_task.name}2{dest_task.name}"
        saved_type, saved_path = None, None
        if model_type is None and path is None:
            saved_type, saved_path = pretrained_transfers.get((src_task.name, dest_task.name), (None, None))

        self.model_type, self.path = model_type or saved_type, path or saved_path
        self.model = None

        if finetuned:
            path = f"{MODELS_DIR}/ft_perceptual/{src_task.name}2{dest_task.name}.pth"
            if os.path.exists(path):
                self.model_type, self.path = saved_type or (lambda: get_model(src_task, dest_task)), path
                print ("Using finetuned: ", path)
                return

        if self.model_type is None:

            if src_task.kind == dest_task.kind and src_task.resize != dest_task.resize:

                class Module(TrainableModel):

                    def __init__(self):
                        super().__init__()

                    def forward(self, x):
                        return resize(x, val=dest_task.resize)

                self.model_type = lambda: Module()
                self.path = None

            path = f"{MODELS_DIR}/{src_task.name}2{dest_task.name}.pth"
            if src_task.name == "keypoints2d" or dest_task.name == "keypoints2d":
                path = f"{MODELS_DIR}/{src_task.name}2{dest_task.name}_new.pth"
            if os.path.exists(path):
                self.model_type, self.path = lambda: get_model(src_task, dest_task), path

        if not pretrained:
            print ("Not using pretrained [heavily discouraged]")
            self.path = None

    def load_model(self):
        if self.model is None:
            if self.path is not None:
                self.model = DataParallelModel.load(self.model_type().to(DEVICE), self.path)
                # if optimizer:
                #     self.model.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
            else:
                self.model = self.model_type().to(DEVICE)
                if isinstance(self.model, nn.Module):
                    self.model = DataParallelModel(self.model)
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
                print(f"")
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

    Transfer('keypoints2d', 'normal', name='k2N'),
    Transfer('normal', 'keypoints2d', name='Nk2'),

    Transfer('sobel_edges', 'reshading', name='Er'),
)

finetuned_transfers = [FineTunedTransfer(transfer) for transfer in functional_transfers]
TRANSFER_MAP = {t.name:t for t in functional_transfers}
functional_transfers = namedtuple('functional_transfers', TRANSFER_MAP.keys())(**TRANSFER_MAP)

def get_transfer_name(transfer):
    for t in functional_transfers:
        if transfer.src_task == t.src_task and transfer.dest_task == t.dest_task:
            return t.name
    return transfer.name

(f, F, g, G, s, S, CE, EC, DE, ED, h, H, n, npstep, RC, k, a, r, d, KC, k3C, Ck3, nr, rn, k3N, Nk3, Er, k2N, N2k) = functional_transfers

if __name__ == "__main__":
    y = g(F(f(x)))
    print (y.shape)






