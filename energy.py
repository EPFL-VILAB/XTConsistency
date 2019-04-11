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
            "ID_subset": dict(
                size=512, 
                realities=("train_subset"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
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
            "ID_subset": dict(
                size=512, 
                realities=("train_subset"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
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
            "ID_subset": dict(
                size=512, 
                realities=("train_subset"),
                paths=[
                    "~x",
                    "n(~x)",
                    "F(RC(~x))",
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
    "rgb2x2normals_plots_size320": {
        "paths": {
            "x": [tasks.rgb(size=320)],
            "~x": [tasks.rgb],
            "~y": [tasks.normal],
            "y^": [tasks.normal(size=320)],
            "n(x)": [tasks.rgb(size=320), tasks.normal(size=320)],
            "principal_curvature": [tasks.rgb(size=320), tasks.principal_curvature(size=320), tasks.normal(size=320)],
            "sobel_edges": [tasks.rgb(size=320), tasks.sobel_edges(size=320), tasks.normal(size=320)],
            "depth_zbuffer": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
            "reshading": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.normal(size=320)],
            "edge_occlusion": [tasks.rgb(size=320), tasks.edge_occlusion(size=320), tasks.normal(size=320)],
            "keypoints3d": [tasks.rgb(size=320), tasks.keypoints3d(size=320), tasks.normal(size=320)],
            "keypoints2d": [tasks.rgb(size=320), tasks.keypoints2d(size=320), tasks.normal(size=320)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
            ],
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature",
                    "sobel_edges",
                    "depth_zbuffer",
                    "reshading",
                    "edge_occlusion",
                    "keypoints3d",
                    "keypoints2d",
                ]
            ),
        },
    },
    "rgb2x2normals_plots_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.normal(size=384)],
            "n(x)": [tasks.rgb(size=384), tasks.normal(size=384)],
            "principal_curvature": [tasks.rgb(size=384), tasks.principal_curvature(size=384), tasks.normal(size=384)],
            "sobel_edges": [tasks.rgb(size=384), tasks.sobel_edges(size=384), tasks.normal(size=384)],
            "depth_zbuffer": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
            "reshading": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.normal(size=384)],
            "edge_occlusion": [tasks.rgb(size=384), tasks.edge_occlusion(size=384), tasks.normal(size=384)],
            "keypoints3d": [tasks.rgb(size=384), tasks.keypoints3d(size=384), tasks.normal(size=384)],
            "keypoints2d": [tasks.rgb(size=384), tasks.keypoints2d(size=384), tasks.normal(size=384)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
            ],
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature",
                    "sobel_edges",
                    "depth_zbuffer",
                    "reshading",
                    "edge_occlusion",
                    "keypoints3d",
                    "keypoints2d",
                ]
            ),
        },
    },
    "rgb2x2normals_plots_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.normal(size=448)],
            "n(x)": [tasks.rgb(size=448), tasks.normal(size=448)],
            "principal_curvature": [tasks.rgb(size=448), tasks.principal_curvature(size=448), tasks.normal(size=448)],
            "sobel_edges": [tasks.rgb(size=448), tasks.sobel_edges(size=448), tasks.normal(size=448)],
            "depth_zbuffer": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448), tasks.normal(size=448)],
            "reshading": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.normal(size=448)],
            "edge_occlusion": [tasks.rgb(size=448), tasks.edge_occlusion(size=448), tasks.normal(size=448)],
            "keypoints3d": [tasks.rgb(size=448), tasks.keypoints3d(size=448), tasks.normal(size=448)],
            "keypoints2d": [tasks.rgb(size=448), tasks.keypoints2d(size=448), tasks.normal(size=448)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
            ],
        },
        "plots": {
            "ID": dict(
                size=448, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature",
                    "sobel_edges",
                    "depth_zbuffer",
                    "reshading",
                    "edge_occlusion",
                    "keypoints3d",
                    "keypoints2d",
                ]
            ),
        },
    },
    "rgb2x2normals_plots_size512": {
        "paths": {
            "x": [tasks.rgb(size=512)],
            "y^": [tasks.normal(size=512)],
            "n(x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "principal_curvature": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
            "sobel_edges": [tasks.rgb(size=512), tasks.sobel_edges(size=512), tasks.normal(size=512)],
            "depth_zbuffer": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512), tasks.normal(size=512)],
            "reshading": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.normal(size=512)],
            "edge_occlusion": [tasks.rgb(size=512), tasks.edge_occlusion(size=512), tasks.normal(size=512)],
            "keypoints3d": [tasks.rgb(size=512), tasks.keypoints3d(size=512), tasks.normal(size=512)],
            "keypoints2d": [tasks.rgb(size=512), tasks.keypoints2d(size=512), tasks.normal(size=512)],
        },
        "losses": {
            ("train", "val"): [
                ("n(x)", "y^"),
            ],
        },
        "plots": {
            "ID": dict(
                size=512, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature",
                    "sobel_edges",
                    "depth_zbuffer",
                    "reshading",
                    "edge_occlusion",
                    "keypoints3d",
                    "keypoints2d",
                ]
            ),
        },
    },
    "consistency_paired_gaussianblur": {
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
            "~n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3), tasks.normal],
            "F(RC(~x))": [tasks.rgb(blur_radius=3), tasks.principal_curvature(blur_radius=3), tasks.normal(blur_radius=3)],
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
}


class EnergyLoss(object):

    def __init__(self, paths, losses, plots, 
        pretrained=True, finetuned=False,
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.metrics = {}

        self.tasks = []
        for realities, losses in self.losses.items():
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
        for realities, losses in self.losses.items():
            if reality in realities:
                for path1, path2 in losses:
                    tasks += [self.paths[path1][0], self.paths[path2][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, realities=[]):

        loss = None
        for reality in realities:
            losses = None
            for realities_l, data in self.losses.items():
                if reality.name in realities_l:
                    losses = data

            path_values = self.compute_paths(graph, 
                paths={
                    path: self.paths[path] for path in \
                        set(path for paths in losses for path in paths)
                    },
                reality=reality)

            self.metrics[reality.name] = defaultdict(list)

            for path1, path2 in losses:
                if self.paths[path1][-1] != self.paths[path2][-1]:
                    raise Exception("Paths have different endpoints.")

                output_task = self.paths[path1][-1]
                path_loss, _ = output_task.norm(path_values[path1], path_values[path2])
                loss = path_loss if loss is None else (path_loss + loss)
                self.metrics[reality.name][path1 + " -> " + path2] += [path_loss.detach().cpu()]

        return loss

    def logger_hooks(self, logger):
        
        name_to_realities = defaultdict(list)
        for realities, losses in self.losses.items():
            for path1, path2 in losses:
                name = path1 + " -> " + path2
                name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)


    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for realities, losses in self.losses.items():
            for path1, path2 in losses:
                name = path1 + " -> " + path2
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
                shape = path_values[list(path_values.keys())[0]].shape
                for i, path in enumerate(paths):
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    images[i].append(X.clamp(min=0, max=1))

            for i in range(0, len(paths)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]",
                resize=config["size"],
            )

    def __repr__(self):
        return str(self.losses)
