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
from task_configs import tasks, get_task, ImageTask
from transfers import functional_transfers, finetuned_transfers, get_transfer_name, Transfer
from datasets import TaskDataset, load_train_val

from matplotlib.cm import get_cmap


import IPython

import pdb

def get_energy_loss(
    config="", mode="winrate",
    pretrained=True, finetuned=True, **kwargs,
):
    """ Loads energy loss from config dict. """
    if isinstance(mode, str):
        mode = {
            "winrate": WinRateEnergyLoss,
        }[mode]
    return mode(**energy_configs[config],
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {

    "multiperceptual_normal": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.reshading],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.reshading],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "nr(y^)": [tasks.normal, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "Nk2(y^)": [tasks.normal, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints2d],
            "Nk3(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
            "nEO(y^)": [tasks.normal, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.normal, tasks.edge_occlusion],
            "imagenet(y^)": [tasks.normal, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.normal, tasks.imagenet],
        },
        "freeze_list": [
            [tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.sobel_edges],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.keypoints3d],
            [tasks.normal, tasks.keypoints2d],
            [tasks.normal, tasks.edge_occlusion],
            [tasks.normal, tasks.imagenet],
        ],
        "losses": {
            "mae": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_reshading": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_reshading": {
                ("train", "val"): [
                    ("d(x)", "reshading"),
                ],
            },
            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_depth_zbuffer": {
                ("train", "val"): [
                    ("r(x)", "depth"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "keypoints2d"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
            "direct_imagenet_percep": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "multiperceptual_reshading": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "n(x)": [tasks.rgb, tasks.reshading],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.normal],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.reshading, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.reshading, tasks.principal_curvature],
            "s(y^)": [tasks.reshading, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.reshading, tasks.sobel_edges],
            "g(y^)": [tasks.reshading, tasks.normal],
            "g(n(x))": [tasks.rgb, tasks.reshading, tasks.normal],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.reshading, tasks.depth_zbuffer],
            "Nk2(y^)": [tasks.reshading, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.reshading, tasks.keypoints2d],
            "Nk3(y^)": [tasks.reshading, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.reshading, tasks.keypoints3d],
            "nEO(y^)": [tasks.reshading, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.reshading, tasks.edge_occlusion],
            "imagenet(y^)": [tasks.reshading, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.reshading, tasks.imagenet],
        },
        "freeze_list": [
            [tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.sobel_edges],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.keypoints3d],
            [tasks.reshading, tasks.keypoints2d],
            [tasks.reshading, tasks.edge_occlusion],
            [tasks.reshading, tasks.imagenet],
        ],
        "losses": {
            "mae": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_normal": {
                ("train", "val"): [
                    ("d(x)", "reshading"),
                ],
            },
            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_depth_zbuffer": {
                ("train", "val"): [
                    ("r(x)", "depth"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "keypoints2d"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
            "direct_imagenet_percep": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                    "depth",
                ]
            ),
        },
    },

    "multiperceptual_depth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.depth_zbuffer],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.reshading],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.normal],
            "reshading": [tasks.reshading],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.principal_curvature],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.normal],
            "g(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.edge_occlusion],
            "imagenet(y^)": [tasks.depth_zbuffer, tasks.imagenet],
            "imagenet(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.imagenet],
        },
        "freeze_list": [
            [tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion],
            [tasks.depth_zbuffer, tasks.imagenet],
        ],
        "losses": {
            "mae": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_normal": {
                ("train", "val"): [
                    ("d(x)", "depth"),
                ],
            },
            "percep_reshading": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_reshading": {
                ("train", "val"): [
                    ("r(x)", "reshading"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "keypoints2d"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
            "percep_imagenet_percep": {
                ("train", "val"): [
                    ("imagenet(n(x))", "imagenet(y^)"),
                ],
            },
            "direct_imagenet_percep": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                ]
            ),
        },
    },
}



def coeff_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1


class EnergyLoss(object):

    def __init__(self, paths, losses, plots,
        pretrained=True, finetuned=False, freeze_list=[]
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
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

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, batch_mean=True):
        loss = {}
        for reality in realities:
            loss_dict = {}
            losses = []
            all_loss_types = set()
            for loss_type, loss_item in self.losses.items():
                all_loss_types.add(loss_type)
                loss_dict[loss_type] = []
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += data
                        if loss_types is not None and loss_type in loss_types:
                            losses += data

            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

            for loss_type, losses in sorted(loss_dict.items()):
                if loss_type not in (loss_types or all_loss_types):
                    continue
                if loss_type not in loss:
                    loss[loss_type] = 0
                for path1, path2 in losses:
                    output_task = self.paths[path1][-1]
                    compute_mask = 'imagenet(n(x))' != path1
                    if "direct" in loss_type:
                        with torch.no_grad():
                            path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mask=compute_mask, compute_mse=False)
                            loss[loss_type] += path_loss
                    else:
                        path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mask=compute_mask, compute_mse=False)
                        loss[loss_type] += path_loss
                        loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                        self.metrics[reality.name][loss_name +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]
                        path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean, compute_mask=compute_mask, compute_mse=True)
                        loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
                        self.metrics[reality.name][loss_name +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

        return loss

    def logger_hooks(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):
                names = [f"{reality}_{name}" for reality in realities]
                if not all(x in data for x in names):
                    return
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)


    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    loss_name = "mae" if "mae" in loss_type else loss_type+"_mae"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)
                    loss_name = "mse" if "mae" in loss_type else loss_type + "_mse"
                    name = loss_name+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                if reality not in self.metrics: continue
                if name not in self.metrics[reality]: continue
                if len(self.metrics[reality][name]) == 0: continue

                logger.update(
                    f"{reality}_{name}",
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}

    def plot_paths(self, graph, logger, realities=[], plot_names=None, epochs=0, tr_step=0,prefix=""):
        error_pairs = {"n(x)": "y^"}
        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]

            realities = config["realities"]
            images = []
            error = False
            cmap = get_cmap("jet")

            first = True
            error_passed_ood = 0
            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])

                shape = list(path_values[list(path_values.keys())[0]].shape)
                shape[1] = 3

                for i, path in enumerate(paths):
                    if path == 'depth': continue
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    if first: images +=[[]]

                    if reality is 'ood' and error_passed_ood==0:
                        images[i].append(X.clamp(min=0, max=1).expand(*shape))
                    elif reality is 'ood' and error_passed_ood==1:
                        images[i+1].append(X.clamp(min=0, max=1).expand(*shape))
                    else:
                        images[-1].append(X.clamp(min=0, max=1).expand(*shape))

                    if path in error_pairs:

                        error = True
                        if first:
                            images += [[]]


                    if error:

                        Y = path_values.get(path, torch.zeros(shape, device=DEVICE))
                        Y_hat = path_values.get(error_pairs[path], torch.zeros(shape, device=DEVICE))

                        out_task = self.paths[path][-1]

                        if self.target_task == "reshading": #Use depth mask
                            Y_mask = path_values.get("depth", torch.zeros(shape, device = DEVICE))
                            mask_task = self.paths["r(x)"][-1]
                            mask = ImageTask.build_mask(Y_mask, val=mask_task.mask_val)
                        else:
                            mask = ImageTask.build_mask(Y_hat, val=out_task.mask_val)

                        errors = ((Y - Y_hat)**2).mean(dim=1, keepdim=True)
                        log_errors = torch.log(errors.clamp(min=0, max=out_task.variance))


                        errors = (3*errors/(out_task.variance)).clamp(min=0, max=1)

                        log_errors = torch.log(errors + 1)
                        log_errors = log_errors / log_errors.max()
                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).expand(*shape).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        if reality is 'ood':
                            images[i+1].append(log_errors)
                            error_passed_ood = 1
                        else:
                            images[-1].append(log_errors)

                        error = False
                first = False

            for i in range(0, len(images)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]",
                resize=config["size"]
            )

    def __repr__(self):
        return str(self.losses)


class WinRateEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}
        self.target_task = kwargs['paths']['y^'][0].name

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print ("percep losses:",self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, compute_grad_ratio=False):

        loss_types = ["mae"] + [("percep_" + loss) for loss in self.percep_losses] + [("direct_" + loss) for loss in self.percep_losses]
        print (self.chosen_losses)
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)

        chosen_percep_mse_losses = [k for k in loss_dict.keys() if 'direct' not in k]
        percep_mse_coeffs = dict.fromkeys(chosen_percep_mse_losses, 1.0)
        ########### to compute loss coefficients #############
        if compute_grad_ratio:
            percep_mse_gradnorms = dict.fromkeys(chosen_percep_mse_losses, 1.0)
            for loss_name in chosen_percep_mse_losses:
                loss_dict[loss_name].mean().backward(retain_graph=True)
                target_weights=list(graph.edge_map[f"('rgb', '{self.target_task}')"].model.parameters())
                percep_mse_gradnorms[loss_name] = sum([l.grad.abs().sum().item() for l in target_weights])/sum([l.numel() for l in target_weights])
                graph.optimizer.zero_grad()
                graph.zero_grad()
                del target_weights
            total_gradnorms = sum(percep_mse_gradnorms.values())
            n_losses = len(chosen_percep_mse_losses)
            for loss_name, val in percep_mse_coeffs.items():
                percep_mse_coeffs[loss_name] = (total_gradnorms-percep_mse_gradnorms[loss_name])/((n_losses-1)*total_gradnorms)
            percep_mse_coeffs["mae"] *= (n_losses-1)
        ###########################################

        for key in self.chosen_losses:
            winrate = torch.mean((loss_dict[f"percep_{key}"] > loss_dict[f"direct_{key}"]).float())
            winrate = winrate.detach().cpu().item()
            if winrate < 1.0:
                self.running_stats[key] = winrate
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"].mean() * percep_mse_coeffs[f"percep_{key}"]
            loss_dict.pop(f"direct_{key}")

        print (self.running_stats)
        loss_dict["mae"] = loss_dict["mae"].mean() * percep_mse_coeffs["mae"]

        return loss_dict, percep_mse_coeffs["mae"]

    def logger_update(self, logger):
        super().logger_update(logger)
        if self.random_select or len(self.running_stats) < len(self.percep_losses):
            self.chosen_losses = random.sample(self.percep_losses, self.k)
        else:
            self.chosen_losses = sorted(self.running_stats, key=self.running_stats.get, reverse=True)[:self.k]

        logger.text (f"Chosen losses: {self.chosen_losses}")


