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
    """ Loads energy loss from config dict. """
    if isinstance(mode, str): 
        mode = {"standard": EnergyLoss}[mode]
    return mode(**energy_configs[config], 
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {
    "two_path_consistency": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
            ],
        },
        "plots": {
            ("test", "ood"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",16
            ]),
        },
    },
    "two_path_consistency_ood": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal(size=256)],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=256)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
                ("F(RC(~x))", "n(~x)")
            ],
        },
        "plots": {
            ("test", "ood"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("test", "ood"): dict(size=512, paths=[
                "~x",
                "n(~x)",
                "F(RC(~x))",
            ]),
        },
    },
    "subset_baseline": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "~y^": [tasks.normal(size=512)],
            "n(x)": [tasks.rgb, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal(size=256)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
            ],
            ("train_subset",): [
                ("n(~x)", "~y^"),
            ],
        },
        "plots": {
            ("test", "ood"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
            ]),
            ("test", "ood"): dict(size=512, paths=[
                "~x",
                "n(~x)",
            ]),
            ("train_subset",): dict(size=512, paths=[
                "~x",
                "n(~x)",
            ]),
        },
    },
    "subset_conservative_baseline": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "~y^": [tasks.normal(size=512)],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal(size=256)],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=256)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
            ],
            ("train_subset",): [
                ("n(~x)", "~y^"),
            ],
        },
        "plots": {
            ("test", "ood"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("test", "ood"): dict(size=512, paths=[
                "~x",
                "n(~x)",
                "F(RC(~x))",
            ]),
            ("train_subset",): dict(size=512, paths=[
                "~x",
                "n(~x)",
                "F(RC(~x))",
            ]),
        },
    },
    "subset_conservative_two_path_ood": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "~y^": [tasks.normal(size=512)],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
                ("F(RC(~x))", "n(~x)"),
            ],
            ("train_subset",): [
                ("n(~x)", "~y^"),
            ],
        },
        "plots": {
            ("test", "ood"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("test", "ood"): dict(size=512, paths=[
                "~x",
                "n(~x)",
                "F(RC(~x))",
            ]),
            ("train_subset",): dict(size=512, paths=[
                "~x",
                "n(~x)",
                "F(RC(~x))",
            ]),
        },
    },
    "consistency_paired_resolution_gt": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "~n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
                ("F(RC(~x))", "n(~x)"),
                ("~n(~x)", "y^"),
            ],
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=512, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
    "consistency_paired_resolution_gt_baseline": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "~n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
                # ("F(RC(~x))", "n(~x)"),
                ("~n(~x)", "y^"),
            ],
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=512, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
    "consistency_paired_resolution_cycle": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "~n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
                ("F(RC(~x))", "n(~x)"),
                ("~n(~x)", "n(x)"),
            ],
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=512, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
    "consistency_paired_resolution_cycle_baseline": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(size=512)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "~n(~x)": [tasks.rgb(size=512), tasks.normal(size=512), tasks.normal],
            "F(RC(~x))": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
                ("F(z^)", "y^"),
                ("RC(x)", "z^"),
                ("F(RC(x))", "y^"),
                ("F(RC(x))", "n(x)"),
                # ("F(RC(~x))", "n(~x)"),
                ("~n(~x)", "n(x)"),
            ],
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=512, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
    "consistency_paired_gaussianblur": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=0)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=0), tasks.normal(blur_radius=0)],
            "F(RC(~x))": [tasks.rgb(blur_radius=0), tasks.principal_curvature(blur_radius=0), tasks.normal(blur_radius=0)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("RC(x)", "z^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                    ("F(RC(~x))", "n(~x)"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=256, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
    "consistency_paired_gaussianblur_cont": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=1)],
            "~~x": [tasks.rgb(blur_radius=3)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=1), tasks.normal(blur_radius=1)],
            "F(RC(~x))": [tasks.rgb(blur_radius=1), tasks.principal_curvature(blur_radius=1), tasks.normal(blur_radius=1)],
            "n(~~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3)],
            "F(RC(~~x))": [tasks.rgb(blur_radius=3), tasks.principal_curvature(blur_radius=3), tasks.normal(blur_radius=3)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("RC(x)", "z^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                    ("F(RC(~x))", "n(~x)"),
                    #("~n(~x)", "n(x)"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=256, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                    "~~x",
                    "n(~~x)",
                    "F(RC(~~x))",
                ]
            ),
        },
    },
    "consistency_paired_gaussianblur_gan": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=3)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3)],
            "F(RC(~x))": [tasks.rgb(blur_radius=3), tasks.principal_curvature(blur_radius=3), tasks.normal(blur_radius=3)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("RC(x)", "z^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                    # ("F(RC(~x))", "n(~x)"),
                ],
            },
            "gan": {
                ("train", "val"): [
                    ("n(x)", "n(~x)"),
                    # ("F(RC(x))", "F(RC(~x))"),
                    #("y^", "n(~x)"),
                    #("y^", "F(RC(~x))"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=256, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
    "consistency_paired_gaussianblur_gan_baseline": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=3)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3)],
            "F(RC(~x))": [tasks.rgb(blur_radius=3), tasks.principal_curvature(blur_radius=3), tasks.normal(blur_radius=3)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("RC(x)", "z^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                    # ("F(RC(~x))", "n(~x)"),
                ],
            },
            "gan": {
                ("train", "val"): [
                    # ("n(x)", "n(~x)"),
                    # ("F(RC(x))", "F(RC(~x))"),
                    #("y^", "n(~x)"),
                    #("y^", "F(RC(~x))"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
                    "z^",
                    "RC(x)",
                ]
            ),
            "OOD": dict(
                size=256, 
                realities=("test", "ood"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
                ]
            ),
        },
    },
}


class EnergyLoss(object):

    def __init__(self, paths, losses, plots,
        pretrained=True, finetuned=False,
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    self.tasks += self.paths[path1] + self.paths[path2]

        for name, config in self.plots.items():
            for path in config["paths"]:
                self.tasks += self.paths[path]
        self.tasks = list(set(self.tasks))

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        path_values = {
            name: graph.sample_path(path, 
                reality=reality, use_cache=True, cache=path_cache,
            ) for name, path in paths.items()
        }
        del path_cache
        return {k: v for k, v in path_values.items() if v is not None}

    def get_tasks(self, reality):
        tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                if reality in realities:
                    for path1, path2 in losses:
                        tasks += [self.paths[path1][0], self.paths[path2][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, discriminator=None, realities=[]):
        loss = {}
        for reality in realities:
            loss_dict = {}
            losses = []
            for loss_type, loss_item in self.losses.items():
                loss_dict[loss_type] = []
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += data
                        losses += data

            path_values = self.compute_paths(graph, 
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            self.metrics[reality.name] = defaultdict(list)

            for loss_type, losses in loss_dict.items():
                if loss_type == 'mse':
                    if 'mse' not in loss:
                        loss['mse'] = 0
                    for path1, path2 in losses:
                        if self.paths[path1][-1] != self.paths[path2][-1]:
                            raise Exception("Paths have different endpoints.")

                        output_task = self.paths[path1][-1]
                        path_loss, _ = output_task.norm(path_values[path1], path_values[path2])
                        loss['mse'] += path_loss
                        self.metrics[reality.name]['mse : '+path1 + " -> " + path2] += [path_loss.detach().cpu()]
                elif loss_type == 'gan':
                    for path1, path2 in losses:
                        if 'gan'+path1+path2 not in loss:
                            loss['disgan'+path1+path2] = 0
                            loss['graphgan'+path1+path2] = 0

                        ## Hack: detach() first pass so only OOD is updated
                        logit_path1 = discriminator[path1+path2](path_values[path1].detach())
                        logit_path2 = discriminator[path1+path2](path_values[path2])
                        binary_label = torch.Tensor([1]*logit_path1.size(0)+[0]*logit_path2.size(0)).float().cuda()
                        gan_loss = nn.BCEWithLogitsLoss(size_average=True)(torch.cat((logit_path1,logit_path2), dim=0).view(-1), binary_label)
                        self.metrics[reality.name]['gan : '+path1 + " -> " + path2] += [gan_loss.detach().cpu()]
                        loss['disgan'+path1+path2] -= gan_loss
                        #binary_label_ood = torch.Tensor([0.5]*(logit_path1.size(0)+logit_path2.size(0))).float().cuda()
                        #gan_loss_ood = nn.BCELoss(size_average=True)(nn.Sigmoid()(torch.cat((logit_path1,logit_path2), dim=0).view(-1)), binary_label_ood)
                        #loss['graphgan'+path1+path2] += gan_loss_ood
                else:
                    raise Exception('Loss {} not implemented.'.format(loss_type)) 

        return loss

    def logger_hooks(self, logger):
        
        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    name = loss_type+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)


    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    name = loss_type+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                logger.update(
                    f"{reality}_{name}", 
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}

    def plot_paths(self, graph, logger, realities=[], prefix=""):
        
        realities_map = {reality.name: reality for reality in realities}
        for name, config in self.plots.items():
            paths = config["paths"]
            realities = config["realities"]
            images = [[] for _ in range(0, len(paths))]
            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])
                shape = list(path_values[list(path_values.keys())[0]].shape)
                shape[1] = 3
                for i, path in enumerate(paths):
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    images[i].append(X.clamp(min=0, max=1).expand(*shape))

            for i in range(0, len(paths)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]",
                resize=config["size"],
            )

    def __repr__(self):
        return str(self.losses)