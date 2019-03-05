import os, sys, math, random, itertools
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from plotting import *
from task_configs import tasks, get_task
from transfers import functional_transfers, finetuned_transfers
from datasets import TaskDataset, load_train_val

import IPython


def get_energy_loss(
    config="", mode="standard",
    pretrained=True, finetuned=True, **kwargs,
):
    if isinstance(mode, str): 
        mode = {"standard": EnergyLoss}[mode]
    return mode(**energy_configs[config], 
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {
    "consistency_two_path": {
        "tasks": [
            tasks.rgb, 
            tasks.normal, 
            tasks.principal_curvature, 
        ],
        "paths": {
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
        ],
    },
    "consistency_two_path_multiresolution": {
        "tasks": [
            tasks.rgb, 
            tasks.normal, 
            tasks.principal_curvature,
            tasks.rgb384,
        ],
        "paths": {
            "n(x)": [tasks.rgb, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "z^": [tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb384, tasks.normal],
            "F(RC(~x))": [tasks.rgb384, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
    },
    "consistency_domain_crop": {
        "tasks": [
            tasks.rgb, 
            tasks.normal, 
            tasks.principal_curvature,
            tasks.rgb512_crop_256,
        ],
        "paths": {
            "n(x)": [tasks.rgb, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "z^": [tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb512_crop_256, tasks.normal],
            "F(RC(~x))": [tasks.rgb512_crop_256, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
    },
}


class EnergyLoss(object):

    def __init__(self, tasks, paths, losses, pretrained=True, finetuned=False):

        self.tasks, self.paths, self.losses = tasks, paths, losses
        self.metrics = {}

    def compute_paths(self, graph, reality=None):
        path_cache = {}
        path_values = {
            name: graph.sample_path(path, 
                reality=reality, use_cache=True, cache=path_cache,
            ) for name, path in self.paths.items()
        }
        return {k: v for k, v in path_values.items() if v is not None}

    def __call__(self, graph, reality=None):

        loss = None
        path_values = self.compute_paths(graph, reality=reality)
        # for path, x in path_values.items():
        #     print (path, x.shape)

        self.metrics[reality] = defaultdict(list)
        for path1, path2 in self.losses:
            if self.paths[path1][-1] != self.paths[path2][-1]:
                raise Exception("Paths have different endpoints.")

            output_task = self.paths[path1][-1]
            path_loss, _ = output_task.norm(path_values[path1], path_values[path2])
            loss = path_loss if loss is None else (path_loss + loss)
            self.metrics[reality][path1 + " -> " + path2] += [path_loss.detach()]

        return loss

    def logger_hooks(self, logger):
        for path1, path2 in self.losses:
            name = path1 + " -> " + path2
            logger.add_hook(
                partial(jointplot, loss_type=f"{name}"), feature=f"val_{name}", freq=1
            )

    def logger_update(self, logger, reality=None):
        for path1, path2 in self.losses:
            name = path1 + " -> " + path2
            logger.update(f"{reality.name}_{name}", torch.mean(torch.stack(self.metrics[reality][name])))

    def plot_paths(self, graph, logger, reality=None):
        path_values = self.compute_paths(graph, reality=reality)
        for path, X in path_values.items():
            output_task = self.paths[path][-1]
            output_task.plot_func(X.clamp(min=0.0, max=1.0), 
                f"{reality.name}_{path}", 
                logger, resize=X.shape[2]
            )

    def __repr__(self):
        return str(self.losses)
