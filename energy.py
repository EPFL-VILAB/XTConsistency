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
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "...": {"test": 128},
        }
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
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 512, "ood": 512},
            "F(RC(~x))": {"test": 512, "ood": 512},
            "...": {"test": 128},
        },
    },
    "consistency_ood_crop": {
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
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 256, "ood": 256},
            "F(RC(~x))": {"test": 256, "ood": 256},
            "...": {"test": 128},
        },
    },
    "consistency_id_384_crop_256_ood_512": {
        "tasks": [
            tasks.rgb384_crop_256, 
            tasks.normal384_crop_256, 
            tasks.principal_curvature384_crop_256,
            tasks.rgb512,
        ],
        "paths": {
            "x": [tasks.rgb384_crop_256],
            "~x": [tasks.rgb512],
            "n(x)": [tasks.rgb384_crop_256, tasks.normal],
            "F(RC(x))": [tasks.rgb384_crop_256, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal384_crop_256],
            "RC(x)": [tasks.rgb384_crop_256, tasks.principal_curvature],
            "z^": [tasks.principal_curvature384_crop_256],
            "F(z^)": [tasks.principal_curvature384_crop_256, tasks.normal],
            "n(~x)": [tasks.rgb512, tasks.normal],
            "F(RC(~x))": [tasks.rgb512, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 512, "ood": 512},
            "F(RC(~x))": {"test": 512, "ood": 512},
            "...": {"test": 128},
        },
    },
    "consistency_id_512_crop_256_ood_640": {
        "tasks": [
            tasks.rgb512_crop_256, 
            tasks.normal512_crop_256, 
            tasks.principal_curvature512_crop_256,
            tasks.rgb512,
        ],
        "paths": {
            "x": [tasks.rgb512_crop_256],
            "~x": [tasks.rgb512],
            "n(x)": [tasks.rgb512_crop_256, tasks.normal],
            "F(RC(x))": [tasks.rgb512_crop_256, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal512_crop_256],
            "RC(x)": [tasks.rgb512_crop_256, tasks.principal_curvature],
            "z^": [tasks.principal_curvature512_crop_256],
            "F(z^)": [tasks.principal_curvature512_crop_256, tasks.normal],
            "n(~x)": [tasks.rgb512, tasks.normal],
            "F(RC(~x))": [tasks.rgb512, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 512, "ood": 512},
            "F(RC(~x))": {"test": 512, "ood": 512},
            "...": {"test": 128},
        },
    },

    "visualize": {
        "tasks": [
            tasks.rgb_domain_shift, 
            tasks.normal_domain_shift, 
            tasks.principal_curvature_domain_shift, 
        ],
        "paths": {
            "RC(x)": [tasks.rgb_domain_shift, tasks.principal_curvature],
            "F(RC(x))": [tasks.rgb_domain_shift, tasks.principal_curvature, tasks.normal],
            "n(x)": [tasks.rgb_domain_shift, tasks.normal],
            "y^": [tasks.normal_domain_shift],
            "z^": [tasks.principal_curvature_domain_shift],
            "F(z^)": [tasks.principal_curvature_domain_shift, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "...": {"test": 128},
        }
    },
    "domain_adaptation": {
        "tasks": [
            tasks.rgb, 
            tasks.normal, 
            tasks.principal_curvature,
            tasks.rgb_domain_shift,
        ],
        "paths": {
            "n(x)": [tasks.rgb, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "z^": [tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb_domain_shift, tasks.normal],
            "F(RC(~x))": [tasks.rgb_domain_shift, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 512, "ood": 512},
            "F(RC(~x))": {"test": 512, "ood": 512},
            "...": {"test": 128},
        },
    },
    "consistency_ood_ds_256_us_512": {
        "tasks": [
            tasks.rgb, 
            tasks.normal, 
            tasks.principal_curvature,
            tasks.rgb256_us512,
        ],
        "paths": {
            "n(x)": [tasks.rgb, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "z^": [tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb256_us512, tasks.normal],
            "F(RC(~x))": [tasks.rgb256_us512, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 512, "ood": 512},
            "F(RC(~x))": {"test": 512, "ood": 512},
            "...": {"test": 128},
        },
    },

    "consistency_id_512_crop_192_ood_512_crop_192": {
        "tasks": [
            tasks.normal, 
            tasks.principal_curvature,
            tasks.rgb512_crop_192, 
            tasks.normal512_crop_192, 
            tasks.principal_curvature512_crop_192,
            # tasks.rgb512_crop_192,
        ],
        "paths": {
            "x": [tasks.rgb512_crop_192],
            "~x": [tasks.rgb512_crop_192],
            "n(x)": [tasks.rgb512_crop_192, tasks.normal],
            "F(RC(x))": [tasks.rgb512_crop_192, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal512_crop_192],
            "RC(x)": [tasks.rgb512_crop_192, tasks.principal_curvature],
            "z^": [tasks.principal_curvature512_crop_192],
            "F(z^)": [tasks.principal_curvature512_crop_192, tasks.normal],
            "n(~x)": [tasks.rgb512_crop_192, tasks.normal],
            "F(RC(~x))": [tasks.rgb512_crop_192, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 256, "ood": 256},
            "F(RC(~x))": {"test": 256, "ood": 256},
            "...": {"test": 128},
        },
    },

    "consistency_id_256_crop_192_ood_512_crop_192": {
        "tasks": [
            tasks.rgb256_crop_192, 
            tasks.normal256_crop_192, 
            tasks.principal_curvature256_crop_192,
            tasks.rgb512_crop_192,
        ],
        "paths": {
            "x": [tasks.rgb256_crop_192],
            "~x": [tasks.rgb512_crop_192],
            "n(x)": [tasks.rgb256_crop_192, tasks.normal],
            "F(RC(x))": [tasks.rgb256_crop_192, tasks.principal_curvature, tasks.normal],
            "y^": [tasks.normal256_crop_192],
            "RC(x)": [tasks.rgb256_crop_192, tasks.principal_curvature],
            "z^": [tasks.principal_curvature256_crop_192],
            "F(z^)": [tasks.principal_curvature256_crop_192, tasks.normal],
            "n(~x)": [tasks.rgb512_crop_192, tasks.normal],
            "F(RC(~x))": [tasks.rgb512_crop_192, tasks.principal_curvature, tasks.normal],
        },
        "losses": [
            ("RC(x)", "z^"),
            ("F(z^)", "y^"),
            ("F(RC(x))", "y^"),
            ("n(x)", "y^"),
            ("F(RC(x))", "n(x)"),
            ("F(RC(~x))", "n(~x)"),
        ],
        "plot_config": {
            "n(x)": {"test": 256, "ood": 256},
            "F(RC(x))": {"test": 256, "ood": 256},
            "F(z^)": {"test": 256},
            "y^": {"test": 256},
            "n(~x)": {"test": 256, "ood": 256},
            "F(RC(~x))": {"test": 256, "ood": 256},
            "...": {"test": 128},
        },
    },
}


class EnergyLoss(object):

    def __init__(self, tasks, paths, losses, 
        pretrained=True, finetuned=False, plot_config=None,
    ):

        self.tasks, self.paths, self.losses = tasks, paths, losses
        self.plot_config = plot_config or {"...": {"test": 256, "ood": 256}}
        if "..." in self.plot_config:
            for path in self.paths:
                self.plot_config[path] = self.plot_config.get(path, 
                    self.plot_config["..."]
                )
        self.plot_config.pop("...")
        self.metrics = {}

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        path_values = {
            name: graph.sample_path(path, 
                reality=reality, use_cache=True, cache=path_cache,
            ) for name, path in paths.items()
        }
        return {k: v for k, v in path_values.items() if v is not None}

    def __call__(self, graph, reality=None):

        loss = None
        path_values = self.compute_paths(graph, reality=reality)
        self.metrics[reality] = defaultdict(list)
        for path1, path2 in self.losses:
            if self.paths[path1][-1].kind != self.paths[path2][-1].kind:
                raise Exception("Paths have different endpoints.")

            output_task = self.paths[path1][-1]
            # if path1 not in path_values or path2 not in path_values:
            #     IPython.embed()
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
            config = self.plot_config.get(path, {})
            if reality.name not in config: continue
            X = path_values[path]
            output_task = self.paths[path][-1]
            output_task.plot_func(X.clamp(min=0.0, max=1.0), 
                f"{reality.name}_{path}", 
                logger, resize=config[reality.name],
            )

    def __repr__(self):
        return str(self.losses)
