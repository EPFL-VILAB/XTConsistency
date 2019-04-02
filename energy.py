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
    "consistency_two_path_ood": {
        "realities": [
            "train",
            "val",
            "ood_consistency",
            "test",
            "ood_test",
            "ood_consistency_test",
        ],
        "tasks": [
            tasks.rgb,
            tasks.normal,
            tasks.principal_curvature,
        ],
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
                ("F(RC(x))", "n(x)")
            ],
            ("ood_consistency",): [
                ("F(RC(x))", "n(x)"),
            ],
        },
        "plots": {
            ("test", "ood_test"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("ood_consistency_test",): dict(size=512, paths=[
                "x",
                "n(x)",
                "F(RC(x))",
            ]),
        },
    },
    "consistency_two_path_ood_subset": {
        "realities": [
            "train",
            "val",
            "train_subset",
            "ood_consistency",
            "test",
            "ood_test",
            "ood_consistency_test",
        ],
        "tasks": [
            tasks.rgb,
            tasks.normal,
            tasks.principal_curvature,
        ],
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
                ("F(RC(x))", "n(x)")
            ],
            ("train_subset",): [
                ("n(x)", "y^"),
            ],
            ("ood_consistency",): [
                ("F(RC(x))", "n(x)"),
            ],
        },
        "plots": {
            ("test", "ood_test"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("ood_consistency_test",): dict(size=512, paths=[
                "x",
                "n(x)",
                "F(RC(x))",
            ]),
        },
    },
    "consistency_baseline_ood_subset": {
        "realities": [
            "train",
            "val",
            "train_subset",
            "ood_consistency",
            "test",
            "ood_test",
            "ood_consistency_test",
        ],
        "tasks": [
            tasks.rgb,
            tasks.normal,
            tasks.principal_curvature,
        ],
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
                # ("F(RC(x))", "n(x)")
            ],
            ("train_subset",): [
                ("n(x)", "y^"),
            ],
            ("ood_consistency",): [
                # ("F(RC(x))", "n(x)"),
            ],
        },
        "plots": {
            ("test", "ood_test"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("ood_consistency_test",): dict(size=512, paths=[
                "x",
                "n(x)",
                "F(RC(x))",
            ]),
        },
    },
    "baseline": {
        "realities": [
            "train",
            "val",
            "train_subset",
            "ood_consistency",
            "test",
            "ood_test",
            "ood_consistency_test",
        ],
        "tasks": [
            tasks.rgb,
            tasks.normal,
            tasks.principal_curvature,
        ],
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
                # ("F(z^)", "y^"),
                # ("RC(x)", "z^"),
                # ("F(RC(x))", "y^"),
                # ("F(RC(x))", "n(x)")
            ],
            ("train_subset",): [
                ("n(x)", "y^"),
            ],
            ("ood_consistency",): [
                # ("F(RC(x))", "n(x)"),
            ],
        },
        "plots": {
            ("test", "ood_test"): dict(size=256, paths=[
                "x",
                "y^",
                "n(x)",
                "F(RC(x))",
                "z^",
                "RC(x)",
            ]),
            ("ood_consistency_test",): dict(size=512, paths=[
                "x",
                "n(x)",
                "F(RC(x))",
            ]),
        },
    },
}


class EnergyLoss(object):

    def __init__(self, realities, tasks, paths, losses, plots, 
        pretrained=True, finetuned=False,
    ):

        self.realities, self.tasks, self.paths, self.losses, self.plots = \
            realities, tasks, paths, losses, plots
        self.metrics = {}

    def load_realities(self, realities):
        assert set(self.realities) == set(reality.name for reality in realities)
        self.realities = {reality.name: reality for reality in realities}

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

    def __call__(self, graph, reality=None):

        loss, losses = None, None
        for realities, data in self.losses.items():
            if reality.name in realities:
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
            self.metrics[reality.name][path1 + " -> " + path2] += [path_loss.detach()]

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

    def plot_paths(self, graph, logger, prefix=""):
        
        for realities, config in self.plots.items():
            paths = config["paths"]
            images = [[] for _ in range(0, len(paths))]
            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=self.realities[reality])
                shape = path_values[list(path_values.keys())[0]].shape
                for i, path in enumerate(paths):
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    images[i].append(X.clamp(min=0, max=1))

            for i in range(0, len(paths)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images, 
                f"{str(tuple(realities))}_{prefix}_[{', '.join(paths)}]",
                resize=config["size"],
            )

    def __repr__(self):
        return str(self.losses)
