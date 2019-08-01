import os, sys, math, random, itertools
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.cm import get_cmap

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from plotting import *
from task_configs import tasks, get_task, ImageTask
from transfers import functional_transfers, finetuned_transfers, get_transfer_name, Transfer
from datasets import TaskDataset, load_train_val

import IPython

import pdb

def get_energy_loss(
    config="", mode="standard",
    pretrained=True, finetuned=True, **kwargs,
):
    """ Loads energy loss from config dict. """
    if isinstance(mode, str): 
        mode = {
            "standard": EnergyLoss,
            "curriculum": CurriculumEnergyLoss,
            "winrate": WinRateEnergyLoss,
        }[mode]
    return mode(**energy_configs[config], 
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {
    "consistency_paired_gaussianblur_gan_patch": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=6)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=6), tasks.normal(blur_radius=6)],
            "F(RC(~x))": [tasks.rgb(blur_radius=6), tasks.principal_curvature(blur_radius=6), tasks.normal(blur_radius=6)],
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
            "gan": {
                ("train", "val"): [
                    ("n(x)", "n(~x)"),
                    ("F(RC(x))", "F(RC(~x))"),
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
    "consistency_multiresolution_gan": {
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
            "gan": {
                ("train", "val"): [
                    ("y^", "n(x)"),
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

    "consistency_multiresolution_gan_gt": {
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
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("RC(x)", "z^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                ],
                ("train_subset",): [
                    ("F(RC(~x))", "n(~x)"),
                ],
            },
            "gan": {
                ("train_subset",): [
                    ("y^", "n(~x)"),
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

    "consistency_paired_gaussianblur_subset": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=6)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=6), tasks.normal(blur_radius=6)],
            "F(RC(~x))": [tasks.rgb(blur_radius=6), tasks.principal_curvature(blur_radius=6), tasks.normal(blur_radius=6)],
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
                ("val",): [
                    ("y^", "n(~x)")
                ]
            },
            "val_ood": {
                ("train_subset",): [
                    ("y^", "n(~x)")
                ]
            },
            "gan": {
                ("train", "val"): [
                    ("y^", "n(~x)"),
                ],
                # ("train_subset",): [
                #     ("y^", "y^"),
                # ],
            }
        },
        "plots": {
            "ID_norm": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
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
            "SUBSET": dict(
                size=256,
                realities=("train_subset",),
                paths=[
                    "~x", 
                    "n(~x)", 
                    "F(RC(~x))", 
                ]
            ),
        },
    },

    "consistency_paired_res_subset": {
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
            "mse_id": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    #  ("RC(x)", "z^"), # fix RC(x)
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                ],
            },
            "mse_ood": {
                ("train", "val"): [
                    ("F(RC(~x))", "n(~x)"),
                ],
                ("val",): [
                    ("~y^", "n(~x)")
                ]
            },
            "val_ood": {
                ("train_subset",): [
                    ("~y^", "n(~x)")
                ]
            },
            "gan": {
                ("train", "val"): [
                    ("~y^", "n(~x)"),
                ],
                # ("train_subset",): [
                #     ("~y^", "~y^"),
                # ],
            }
        },
        "plots": {
            "ID_norm": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
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
            "SUBSET": dict(
                size=256, 
                realities=("train_subset",),
                paths=[
                    "~x", 
                    "n(~x)", 
                    "F(RC(~x))", 
                ]
            ),
        },
    },


    "consistency_gaussianblur": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=6)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=6), tasks.normal(blur_radius=6)],
            "F(RC(~x))": [tasks.rgb(blur_radius=6), tasks.principal_curvature(blur_radius=6), tasks.normal(blur_radius=6)],
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
                ("val",): [
                    ("y^", "n(~x)")
                ]
            },
        },
        "plots": {
            "ID_norm": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "F(RC(x))",
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

    "baseline": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                ]
            ),
        },
    },

    "percep_curv": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints": [tasks.keypoints3d],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk2(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
        },
        "freeze_list": [
            [tasks.normal, tasks.sobel_edges],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "indirect_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
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
                    "s(y^)",
                    "s(n(x))",
                ]
            ),
        },
    },

    "percep_edge": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints": [tasks.keypoints3d],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk2(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
        },
        "freeze_list": [
            [tasks.normal, tasks.sobel_edges],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "a(x)"),  #  > WINRATE( ,<d(x), GT>)
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("s(y^)", "a(x)"),
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
                    "s(y^)",
                    "s(n(x))",
                ]
            ),
        },
    },

    "percep_reshading": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints": [tasks.keypoints3d],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk2(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
        },
        "freeze_list": [
            [tasks.normal, tasks.reshading],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "nr(y^)",
                    "nr(n(x))",
                ]
            ),
        },
    },

    "percep_keypoints3d": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints": [tasks.keypoints3d],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk3(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
        },
        "freeze_list": [
            [tasks.normal, tasks.keypoints3d],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_keypoints": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints": {
                ("train", "val"): [
                    ("k(x)", "keypoints"),
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
                    "Nk3(y^)",
                    "Nk3(n(x))",
                ]
            ),
        },
    },

    "multiperceptual": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints3d": [tasks.keypoints3d],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk3(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
        },
        "freeze_list": [
            [tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.sobel_edges],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.keypoints3d],
        ],
        "losses": {
            "mse": {
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
            "indirect_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("s(y^)", "edge"),
                ],
            },
            "indirect_edge": {
                ("train", "val"): [
                    ("s(n(x))", "edge"),
                ],
            },
            "percep_depth": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_depth": {
                ("train", "val"): [
                    ("d(x)", "depth"),
                ],
            },
            "indirect_depth": {
                ("train", "val"): [
                    ("g(n(x))", "depth"),
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
            "indirect_reshading": {
                ("train", "val"): [
                    ("nr(n(x))", "reshading"),
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
            "indirect_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "keypoints3d"),
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
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(y^)",
                    "nr(n(x))",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                ]
            ),
        },
    },

    # "multiperceptual_depth": {
    #     "paths": {
    #         "x": [tasks.rgb],
    #         "y^": [tasks.depth_zbuffer],
    #         "n(x)": [tasks.rgb, tasks.depth_zbuffer],
    #         "RC(x)": [tasks.rgb, tasks.principal_curvature],
    #         "a(x)": [tasks.rgb, tasks.sobel_edges],
    #         "d(x)": [tasks.rgb, tasks.normal],
    #         "r(x)": [tasks.rgb, tasks.reshading],
    #         "k(x)": [tasks.rgb, tasks.keypoints3d],
    #         "curv": [tasks.principal_curvature],
    #         "edge": [tasks.sobel_edges],
    #         "depth": [tasks.normal],
    #         "reshading": [tasks.reshading],
    #         "keypoints": [tasks.keypoints3d],
    #         "f(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
    #         "f(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.principal_curvature],
    #         "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
    #         "s(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.sobel_edges],
    #         "g(y^)": [tasks.depth_zbuffer, tasks.normal],
    #         "g(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
    #         "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
    #         "nr(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.reshading],
    #         "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
    #         "Nk2(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints3d],
    #     },
    #     "freeze_list": [
    #         [tasks.depth_zbuffer, tasks.principal_curvature],
    #         [tasks.depth_zbuffer, tasks.sobel_edges],
    #         [tasks.depth_zbuffer, tasks.normal],
    #         [tasks.depth_zbuffer, tasks.reshading],
    #         [tasks.depth_zbuffer, tasks.keypoints3d],
    #     ],
    #     "losses": {
    #         "mse": {
    #             ("train", "val"): [
    #                 ("n(x)", "y^"),
    #             ],
    #         },
    #         "percep_curv": {
    #             ("train", "val"): [
    #                 ("f(n(x))", "f(y^)"),
    #             ],
    #         },
    #         "direct_curv": {
    #             ("train", "val"): [
    #                 ("RC(x)", "curv"),
    #             ],
    #         },
    #         "percep_edge": {
    #             ("train", "val"): [
    #                 ("s(n(x))", "s(y^)"),
    #             ],
    #         },
    #         "direct_edge": {
    #             ("train", "val"): [
    #                 ("a(x)", "s(y^)"),
    #             ],
    #         },
    #         "percep_normal": {
    #             ("train", "val"): [
    #                 ("g(n(x))", "g(y^)"),
    #             ],
    #         },
    #         "direct_normal": {
    #             ("train", "val"): [
    #                 ("d(x)", "depth"),
    #             ],
    #         },
    #         "percep_reshading": {
    #             ("train", "val"): [
    #                 ("nr(n(x))", "nr(y^)"),
    #             ],
    #         },
    #         "direct_reshading": {
    #             ("train", "val"): [
    #                 ("r(x)", "reshading"),
    #             ],
    #         },
    #         "percep_keypoints": {
    #             ("train", "val"): [
    #                 ("Nk2(n(x))", "Nk2(y^)"),
    #             ],
    #         },
    #         "direct_keypoints": {
    #             ("train", "val"): [
    #                 ("k(x)", "keypoints"),
    #             ],
    #         },
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=256, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "f(y^)",
    #                 "f(n(x))",
    #                 "s(y^)",
    #                 "s(n(x))",
    #                 "g(y^)",
    #                 "g(n(x))",
    #                 "nr(n(x))",
    #                 "nr(y^)",
    #                 "Nk2(y^)",
    #                 "Nk2(n(x))",
    #             ]
    #         ),
    #     },
    # },


    # "multiperceptual_reshading": {
    #     "paths": {
    #         "x": [tasks.rgb],
    #         "y^": [tasks.normal],
    #         "n(x)": [tasks.rgb, tasks.reshading],
    #         "RC(x)": [tasks.rgb, tasks.principal_curvature],
    #         "a(x)": [tasks.rgb, tasks.sobel_edges],
    #         "d(x)": [tasks.rgb, tasks.depth_zbuffer],
    #         "r(x)": [tasks.rgb, tasks.normal],
    #         "k(x)": [tasks.rgb, tasks.keypoints3d],
    #         "curv": [tasks.principal_curvature],
    #         "edge": [tasks.sobel_edges],
    #         "depth": [tasks.depth_zbuffer],
    #         "reshading": [tasks.normal],
    #         "keypoints": [tasks.keypoints3d],
    #         "f(y^)": [tasks.reshading, tasks.principal_curvature],
    #         "f(n(x))": [tasks.rgb, tasks.reshading, tasks.principal_curvature],
    #         "s(y^)": [tasks.reshading, tasks.sobel_edges],
    #         "s(n(x))": [tasks.rgb, tasks.reshading, tasks.sobel_edges],
    #         "g(y^)": [tasks.reshading, tasks.depth_zbuffer],
    #         "g(n(x))": [tasks.rgb, tasks.reshading, tasks.depth_zbuffer],
    #         "nr(y^)": [tasks.reshading, tasks.normal],
    #         "nr(n(x))": [tasks.rgb, tasks.reshading, tasks.normal],
    #         "Nk2(y^)": [tasks.reshading, tasks.keypoints3d],
    #         "Nk2(n(x))": [tasks.rgb, tasks.reshading, tasks.keypoints3d],
    #     },
    #     "freeze_list": [
    #         [tasks.reshading, tasks.principal_curvature],
    #         [tasks.reshading, tasks.sobel_edges],
    #         [tasks.reshading, tasks.depth_zbuffer],
    #         [tasks.reshading, tasks.normal],
    #         [tasks.reshading, tasks.keypoints3d],
    #     ],
    #     "losses": {
    #         "mse": {
    #             ("train", "val"): [
    #                 ("n(x)", "y^"),
    #             ],
    #         },
    #         "percep_curv": {
    #             ("train", "val"): [
    #                 ("f(n(x))", "f(y^)"),
    #             ],
    #         },
    #         "direct_curv": {
    #             ("train", "val"): [
    #                 ("RC(x)", "curv"),
    #             ],
    #         },
    #         "percep_edge": {
    #             ("train", "val"): [
    #                 ("s(n(x))", "s(y^)"),
    #             ],
    #         },
    #         "direct_edge": {
    #             ("train", "val"): [
    #                 ("a(x)", "s(y^)"),
    #             ],
    #         },
    #         "percep_depth": {
    #             ("train", "val"): [
    #                 ("g(n(x))", "g(y^)"),
    #             ],
    #         },
    #         "direct_depth": {
    #             ("train", "val"): [
    #                 ("d(x)", "depth"),
    #             ],
    #         },
    #         "percep_normal": {
    #             ("train", "val"): [
    #                 ("nr(n(x))", "nr(y^)"),
    #             ],
    #         },
    #         "direct_normal": {
    #             ("train", "val"): [
    #                 ("r(x)", "reshading"),
    #             ],
    #         },
    #         "percep_keypoints": {
    #             ("train", "val"): [
    #                 ("Nk2(n(x))", "Nk2(y^)"),
    #             ],
    #         },
    #         "direct_keypoints": {
    #             ("train", "val"): [
    #                 ("k(x)", "keypoints"),
    #             ],
    #         },
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=256, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "f(y^)",
    #                 "f(n(x))",
    #                 "s(y^)",
    #                 "s(n(x))",
    #                 "g(y^)",
    #                 "g(n(x))",
    #                 "nr(n(x))",
    #                 "nr(y^)",
    #                 "Nk2(y^)",
    #                 "Nk2(n(x))",
    #             ]
    #         ),
    #     },
    # },

    "multiperceptual_expanded": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
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
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk2(y^)": [tasks.normal, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints2d],
            "Nk3(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
            "nEO(y^)": [tasks.normal, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.normal, tasks.edge_occlusion],
        },
        "freeze_list": [
            [tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.sobel_edges],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.keypoints2d],
            [tasks.normal, tasks.keypoints3d],
            [tasks.normal, tasks.edge_occlusion],
        ],
        "losses": {
            "mse": {
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
            "indirect_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("s(y^)", "edge"),
                ],
            },
            "indirect_edge": {
                ("train", "val"): [
                    ("s(n(x))", "edge"),
                ],
            },
            "percep_depth": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_depth": {
                ("train", "val"): [
                    ("d(x)", "depth"),
                ],
            },
            "indirect_depth": {
                ("train", "val"): [
                    ("g(n(x))", "depth"),
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
            "indirect_reshading": {
                ("train", "val"): [
                    ("nr(n(x))", "reshading"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(y^)", "keypoints2d"),
                ],
            },
            "indirect_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "keypoints2d"),
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
            "indirect_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "keypoints3d"),
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
            "indirect_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "edge_occlusion"),
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
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                ]
            ),
        },
    },
    "multiperceptual_expanded_depth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.depth_zbuffer],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
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
        },
        "freeze_list": [
            [tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.edge_occlusion],
        ],
        "losses": {
            "mse": {
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
            "indirect_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("s(y^)", "edge"),
                ],
            },
            "indirect_edge": {
                ("train", "val"): [
                    ("s(n(x))", "edge"),
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
            "indirect_normal": {
                ("train", "val"): [
                    ("g(n(x))", "depth"),
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
            "indirect_reshading": {
                ("train", "val"): [
                    ("nr(n(x))", "reshading"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(y^)", "keypoints2d"),
                ],
            },
            "indirect_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "keypoints2d"),
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
            "indirect_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "keypoints3d"),
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
            "indirect_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "edge_occlusion"),
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
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                ]
            ),
        },
    },

    "cross_perceptual": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("f(y^)", "z^"),
                ],
            },
            "percep": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                    ("f(n(x))", "z^"),
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
                    "f(y^)",
                    "f(n(x))",
                ]
            ),
        },
    },

    "conservative": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "w^": [tasks.sobel_edges],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "S(w^)": [tasks.sobel_edges, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "S(a(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
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
                ]
            ),
        },
    },

    "doubletriangle_conservative": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "w^": [tasks.sobel_edges],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "S(w^)": [tasks.sobel_edges, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "S(a(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal],
        },
        "losses": {
            "triangle1_mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("F(z^)", "y^"),
                    ("F(RC(x))", "y^"),
                    ("F(RC(x))", "n(x)"),
                ],
            },
            "triangle2_mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                    ("S(w^)", "y^"),
                    ("S(a(x))", "y^"),
                    ("S(a(x))", "n(x)"),
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
                    "S(a(x))",
                ]
            ),
        },
    },

    "doubletriangle_perceptual": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "w^": [tasks.sobel_edges],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "S(w^)": [tasks.sobel_edges, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "S(a(x))": [tasks.rgb, tasks.sobel_edges, tasks.normal],
        },
        "losses": {
            "triangle1_mse": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "triangle2_mse": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
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
                    "S(a(x))",
                ]
            ),
        },
    },

    "rgb2x2normals_plots": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb],
            "~y": [tasks.normal],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "principal_curvature2normal": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "sobel_edges2normal": [tasks.rgb, tasks.sobel_edges, tasks.normal],
            "depth_zbuffer2normal": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
            "reshading2normal": [tasks.rgb, tasks.reshading, tasks.normal],
            "edge_occlusion2normal": [tasks.rgb, tasks.edge_occlusion, tasks.normal],
            "keypoints3d2normal": [tasks.rgb, tasks.keypoints3d, tasks.normal],
            "keypoints2d2normal": [tasks.rgb, tasks.keypoints2d, tasks.normal],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
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
            "principal_curvature2normal": [tasks.rgb(size=320), tasks.principal_curvature(size=320), tasks.normal(size=320)],
            "sobel_edges2normal": [tasks.rgb(size=320), tasks.sobel_edges(size=320), tasks.normal(size=320)],
            "depth_zbuffer2normal": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
            "reshading2normal": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.normal(size=320)],
            "edge_occlusion2normal": [tasks.rgb(size=320), tasks.edge_occlusion(size=320), tasks.normal(size=320)],
            "keypoints3d2normal": [tasks.rgb(size=320), tasks.keypoints3d(size=320), tasks.normal(size=320)],
            "keypoints2d2normal": [tasks.rgb(size=320), tasks.keypoints2d(size=320), tasks.normal(size=320)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "rgb2x2normals_plots_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.normal(size=384)],
            "n(x)": [tasks.rgb(size=384), tasks.normal(size=384)],
            "principal_curvature2normal": [tasks.rgb(size=384), tasks.principal_curvature(size=384), tasks.normal(size=384)],
            "sobel_edges2normal": [tasks.rgb(size=384), tasks.sobel_edges(size=384), tasks.normal(size=384)],
            "depth_zbuffer2normal": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
            "reshading2normal": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.normal(size=384)],
            "edge_occlusion2normal": [tasks.rgb(size=384), tasks.edge_occlusion(size=384), tasks.normal(size=384)],
            "keypoints3d2normal": [tasks.rgb(size=384), tasks.keypoints3d(size=384), tasks.normal(size=384)],
            "keypoints2d2normal": [tasks.rgb(size=384), tasks.keypoints2d(size=384), tasks.normal(size=384)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "rgb2x2normals_plots_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.normal(size=448)],
            "n(x)": [tasks.rgb(size=448), tasks.normal(size=448)],
            "principal_curvature2normal": [tasks.rgb(size=448), tasks.principal_curvature(size=448), tasks.normal(size=448)],
            "sobel_edges2normal": [tasks.rgb(size=448), tasks.sobel_edges(size=448), tasks.normal(size=448)],
            "depth_zbuffer2normal": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448), tasks.normal(size=448)],
            "reshading2normal": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.normal(size=448)],
            "edge_occlusion2normal": [tasks.rgb(size=448), tasks.edge_occlusion(size=448), tasks.normal(size=448)],
            "keypoints3d2normal": [tasks.rgb(size=448), tasks.keypoints3d(size=448), tasks.normal(size=448)],
            "keypoints2d2normal": [tasks.rgb(size=448), tasks.keypoints2d(size=448), tasks.normal(size=448)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=448, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "rgb2x2normals_plots_size512": {
        "paths": {
            "x": [tasks.rgb(size=512)],
            "y^": [tasks.normal(size=512)],
            "n(x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "principal_curvature2normal": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
            "sobel_edges2normal": [tasks.rgb(size=512), tasks.sobel_edges(size=512), tasks.normal(size=512)],
            "depth_zbuffer2normal": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512), tasks.normal(size=512)],
            "reshading2normal": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.normal(size=512)],
            "edge_occlusion2normal": [tasks.rgb(size=512), tasks.edge_occlusion(size=512), tasks.normal(size=512)],
            "keypoints3d2normal": [tasks.rgb(size=512), tasks.keypoints3d(size=512), tasks.normal(size=512)],
            "keypoints2d2normal": [tasks.rgb(size=512), tasks.keypoints2d(size=512), tasks.normal(size=512)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=512, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "y2normals_plots": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "principal_curvature2normal": [tasks.principal_curvature, tasks.normal],
            "sobel_edges2normal": [tasks.sobel_edges, tasks.normal],
            "depth_zbuffer2normal": [tasks.depth_zbuffer, tasks.normal],
            "reshading2normal": [tasks.reshading, tasks.normal],
            "edge_occlusion2normal": [tasks.edge_occlusion, tasks.normal],
            "keypoints3d2normal": [tasks.keypoints3d, tasks.normal],
            "keypoints2d2normal": [tasks.keypoints2d, tasks.normal],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test",), 
                paths=[
                    "x",
                    "y^",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "y2normals_plots_size320": {
        "paths": {
            "x": [tasks.rgb(size=320)],
            "y^": [tasks.normal(size=320)],
            "principal_curvature2normal": [tasks.principal_curvature(size=320), tasks.normal(size=320)],
            "sobel_edges2normal": [tasks.sobel_edges(size=320), tasks.normal(size=320)],
            "depth_zbuffer2normal": [tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
            "reshading2normal": [tasks.reshading(size=320), tasks.normal(size=320)],
            "edge_occlusion2normal": [tasks.edge_occlusion(size=320), tasks.normal(size=320)],
            "keypoints3d2normal": [tasks.keypoints3d(size=320), tasks.normal(size=320)],
            "keypoints2d2normal": [tasks.keypoints2d(size=320), tasks.normal(size=320)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test",), 
                paths=[
                    "x",
                    "y^",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "y2normals_plots_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.normal(size=384)],
            "principal_curvature2normal": [tasks.principal_curvature(size=384), tasks.normal(size=384)],
            "sobel_edges2normal": [tasks.sobel_edges(size=384), tasks.normal(size=384)],
            "depth_zbuffer2normal": [tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
            "reshading2normal": [tasks.reshading(size=384), tasks.normal(size=384)],
            "edge_occlusion2normal": [tasks.edge_occlusion(size=384), tasks.normal(size=384)],
            "keypoints3d2normal": [tasks.keypoints3d(size=384), tasks.normal(size=384)],
            "keypoints2d2normal": [tasks.keypoints2d(size=384), tasks.normal(size=384)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test",), 
                paths=[
                    "x",
                    "y^",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "y2normals_plots_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.normal(size=448)],
            "principal_curvature2normal": [tasks.principal_curvature(size=448), tasks.normal(size=448)],
            "sobel_edges2normal": [tasks.sobel_edges(size=448), tasks.normal(size=448)],
            "depth_zbuffer2normal": [tasks.depth_zbuffer(size=448), tasks.normal(size=448)],
            "reshading2normal": [tasks.reshading(size=448), tasks.normal(size=448)],
            "edge_occlusion2normal": [tasks.edge_occlusion(size=448), tasks.normal(size=448)],
            "keypoints3d2normal": [tasks.keypoints3d(size=448), tasks.normal(size=448)],
            "keypoints2d2normal": [tasks.keypoints2d(size=448), tasks.normal(size=448)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=448, 
                realities=("test",), 
                paths=[
                    "x",
                    "y^",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "y2normals_plots_size512": {
        "paths": {
            "x": [tasks.rgb(size=512)],
            "y^": [tasks.normal(size=512)],
            "principal_curvature2normal": [tasks.principal_curvature(size=512), tasks.normal(size=512)],
            "sobel_edges2normal": [tasks.sobel_edges(size=512), tasks.normal(size=512)],
            "depth_zbuffer2normal": [tasks.depth_zbuffer(size=512), tasks.normal(size=512)],
            "reshading2normal": [tasks.reshading(size=512), tasks.normal(size=512)],
            "edge_occlusion2normal": [tasks.edge_occlusion(size=512), tasks.normal(size=512)],
            "keypoints3d2normal": [tasks.keypoints3d(size=512), tasks.normal(size=512)],
            "keypoints2d2normal": [tasks.keypoints2d(size=512), tasks.normal(size=512)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=512, 
                realities=("test",), 
                paths=[
                    "x",
                    "y^",
                    "principal_curvature2normal",
                    "sobel_edges2normal",
                    "depth_zbuffer2normal",
                    "reshading2normal",
                    "edge_occlusion2normal",
                    "keypoints3d2normal",
                    "keypoints2d2normal",
                ]
            ),
        },
    },
    "rgb2x_plots": {
        "paths": {
            "x": [tasks.rgb],
            "rgb2normal": [tasks.rgb, tasks.normal],
            "rgb2principal_curvature": [tasks.rgb, tasks.principal_curvature],
            "rgb2sobel_edges": [tasks.rgb, tasks.sobel_edges],
            "rgb2depth_zbuffer": [tasks.rgb, tasks.depth_zbuffer],
            "rgb2reshading": [tasks.rgb, tasks.reshading],
            "rgb2edge_occlusion": [tasks.rgb, tasks.edge_occlusion],
            "rgb2keypoints3d": [tasks.rgb, tasks.keypoints3d],
            "rgb2keypoints2d": [tasks.rgb, tasks.keypoints2d],
            "normal": [tasks.normal],
            "principal_curvature": [tasks.principal_curvature],
            "sobel_edges": [tasks.sobel_edges],
            "depth_zbuffer": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "edge_occlusion": [tasks.edge_occlusion],
            "keypoints3d": [tasks.keypoints3d],
            "keypoints2d": [tasks.keypoints2d],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "normal",
                    "rgb2normal",
                    "principal_curvature",
                    "rgb2principal_curvature",
                    "sobel_edges",
                    "rgb2sobel_edges",
                    "depth_zbuffer",
                    "rgb2depth_zbuffer",
                    "reshading",
                    "rgb2reshading",
                    "edge_occlusion",
                    "rgb2edge_occlusion",
                    "keypoints3d",
                    "rgb2keypoints3d",
                    "keypoints2d",
                    "rgb2keypoints2d",
                ]
            ),
        },
    },
    "rgb2x_plots_size320": {
        "paths": {
            "x": [tasks.rgb(size=320)],
            "y^": [tasks.normal(size=320)],
            "rgb2normal": [tasks.rgb(size=320), tasks.normal(size=320)],
            "rgb2principal_curvature": [tasks.rgb(size=320), tasks.principal_curvature(size=320)],
            "rgb2sobel_edges": [tasks.rgb(size=320), tasks.sobel_edges(size=320)],
            "rgb2depth_zbuffer": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320)],
            "rgb2reshading": [tasks.rgb(size=320), tasks.reshading(size=320)],
            "rgb2edge_occlusion": [tasks.rgb(size=320), tasks.edge_occlusion(size=320)],
            "rgb2keypoints3d": [tasks.rgb(size=320), tasks.keypoints3d(size=320)],
            "rgb2keypoints2d": [tasks.rgb(size=320), tasks.keypoints2d(size=320)],
            "normal": [tasks.normal(size=320)],
            "principal_curvature": [tasks.principal_curvature(size=320)],
            "sobel_edges": [tasks.sobel_edges(size=320)],
            "depth_zbuffer": [tasks.depth_zbuffer(size=320)],
            "reshading": [tasks.reshading(size=320)],
            "edge_occlusion": [tasks.edge_occlusion(size=320)],
            "keypoints3d": [tasks.keypoints3d(size=320)],
            "keypoints2d": [tasks.keypoints2d(size=320)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "normal",
                    "rgb2normal",
                    "principal_curvature",
                    "rgb2principal_curvature",
                    "sobel_edges",
                    "rgb2sobel_edges",
                    "depth_zbuffer",
                    "rgb2depth_zbuffer",
                    "reshading",
                    "rgb2reshading",
                    "edge_occlusion",
                    "rgb2edge_occlusion",
                    "keypoints3d",
                    "rgb2keypoints3d",
                    "keypoints2d",
                    "rgb2keypoints2d",
                ]
            ),
        },
    },
    "rgb2x_plots_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.normal(size=384)],
            "rgb2normal": [tasks.rgb(size=384), tasks.normal(size=384)],
            "rgb2principal_curvature": [tasks.rgb(size=384), tasks.principal_curvature(size=384)],
            "rgb2sobel_edges": [tasks.rgb(size=384), tasks.sobel_edges(size=384)],
            "rgb2depth_zbuffer": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384)],
            "rgb2reshading": [tasks.rgb(size=384), tasks.reshading(size=384)],
            "rgb2edge_occlusion": [tasks.rgb(size=384), tasks.edge_occlusion(size=384)],
            "rgb2keypoints3d": [tasks.rgb(size=384), tasks.keypoints3d(size=384)],
            "rgb2keypoints2d": [tasks.rgb(size=384), tasks.keypoints2d(size=384)],
            "normal": [tasks.normal(size=384)],
            "principal_curvature": [tasks.principal_curvature(size=384)],
            "sobel_edges": [tasks.sobel_edges(size=384)],
            "depth_zbuffer": [tasks.depth_zbuffer(size=384)],
            "reshading": [tasks.reshading(size=384)],
            "edge_occlusion": [tasks.edge_occlusion(size=384)],
            "keypoints3d": [tasks.keypoints3d(size=384)],
            "keypoints2d": [tasks.keypoints2d(size=384)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "normal",
                    "rgb2normal",
                    "principal_curvature",
                    "rgb2principal_curvature",
                    "sobel_edges",
                    "rgb2sobel_edges",
                    "depth_zbuffer",
                    "rgb2depth_zbuffer",
                    "reshading",
                    "rgb2reshading",
                    "edge_occlusion",
                    "rgb2edge_occlusion",
                    "keypoints3d",
                    "rgb2keypoints3d",
                    "keypoints2d",
                    "rgb2keypoints2d",
                ]
            ),
        },
    },
    "rgb2x_plots_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.normal(size=448)],
            "rgb2normal": [tasks.rgb(size=448), tasks.normal(size=448)],
            "rgb2principal_curvature": [tasks.rgb(size=448), tasks.principal_curvature(size=448)],
            "rgb2sobel_edges": [tasks.rgb(size=448), tasks.sobel_edges(size=448)],
            "rgb2depth_zbuffer": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448)],
            "rgb2reshading": [tasks.rgb(size=448), tasks.reshading(size=448)],
            "rgb2edge_occlusion": [tasks.rgb(size=448), tasks.edge_occlusion(size=448)],
            "rgb2keypoints3d": [tasks.rgb(size=448), tasks.keypoints3d(size=448)],
            "rgb2keypoints2d": [tasks.rgb(size=448), tasks.keypoints2d(size=448)],
            "normal": [tasks.normal(size=448)],
            "principal_curvature": [tasks.principal_curvature(size=448)],
            "sobel_edges": [tasks.sobel_edges(size=448)],
            "depth_zbuffer": [tasks.depth_zbuffer(size=448)],
            "reshading": [tasks.reshading(size=448)],
            "edge_occlusion": [tasks.edge_occlusion(size=448)],
            "keypoints3d": [tasks.keypoints3d(size=448)],
            "keypoints2d": [tasks.keypoints2d(size=448)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=448, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "normal",
                    "rgb2normal",
                    "principal_curvature",
                    "rgb2principal_curvature",
                    "sobel_edges",
                    "rgb2sobel_edges",
                    "depth_zbuffer",
                    "rgb2depth_zbuffer",
                    "reshading",
                    "rgb2reshading",
                    "edge_occlusion",
                    "rgb2edge_occlusion",
                    "keypoints3d",
                    "rgb2keypoints3d",
                    "keypoints2d",
                    "rgb2keypoints2d",
                ]
            ),
        },
    },
    "rgb2x_plots_size512": {
        "paths": {
            "x": [tasks.rgb(size=512)],
            "y^": [tasks.normal(size=512)],
            "rgb2normal": [tasks.rgb(size=512), tasks.normal(size=512)],
            "rgb2principal_curvature": [tasks.rgb(size=512), tasks.principal_curvature(size=512)],
            "rgb2sobel_edges": [tasks.rgb(size=512), tasks.sobel_edges(size=512)],
            "rgb2depth_zbuffer": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512)],
            "rgb2reshading": [tasks.rgb(size=512), tasks.reshading(size=512)],
            "rgb2edge_occlusion": [tasks.rgb(size=512), tasks.edge_occlusion(size=512)],
            "rgb2keypoints3d": [tasks.rgb(size=512), tasks.keypoints3d(size=512)],
            "rgb2keypoints2d": [tasks.rgb(size=512), tasks.keypoints2d(size=512)],
            "normal": [tasks.normal(size=512)],
            "principal_curvature": [tasks.principal_curvature(size=512)],
            "sobel_edges": [tasks.sobel_edges(size=512)],
            "depth_zbuffer": [tasks.depth_zbuffer(size=512)],
            "reshading": [tasks.reshading(size=512)],
            "edge_occlusion": [tasks.edge_occlusion(size=512)],
            "keypoints3d": [tasks.keypoints3d(size=512)],
            "keypoints2d": [tasks.keypoints2d(size=512)],
        },
        "losses": {
        },
        "plots": {
            "ID": dict(
                size=512, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "normal",
                    "rgb2normal",
                    "principal_curvature",
                    "rgb2principal_curvature",
                    "sobel_edges",
                    "rgb2sobel_edges",
                    "depth_zbuffer",
                    "rgb2depth_zbuffer",
                    "reshading",
                    "rgb2reshading",
                    "edge_occlusion",
                    "rgb2edge_occlusion",
                    "keypoints3d",
                    "rgb2keypoints3d",
                    "keypoints2d",
                    "rgb2keypoints2d",
                ]
            ),
        },
    },


    "baseline_reshade_size256": {
        "paths": {
            "x": [tasks.rgb(size=256)],
            "y^": [tasks.reshading(size=256)],
            "n(x)": [tasks.rgb(size=256), tasks.reshading(size=256)],
            "RC(x)": [tasks.rgb(size=256), tasks.principal_curvature(size=256)],
            "a(x)": [tasks.rgb(size=256), tasks.sobel_edges(size=256)],
            "d(x)": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256)],
            "r(x)": [tasks.rgb(size=256), tasks.normal(size=256)],
            "k2(x)": [tasks.rgb(size=256), tasks.keypoints2d(size=256)],
            "k3(x)": [tasks.rgb(size=256), tasks.keypoints3d(size=256)],
            "EO(x)": [tasks.rgb(size=256), tasks.edge_occlusion(size=256)],
            "curv": [tasks.principal_curvature(size=256)],
            "edge": [tasks.sobel_edges(size=256)],
            "depth": [tasks.depth_zbuffer(size=256)],
            "reshading": [tasks.normal(size=256)],
            "keypoints2d": [tasks.keypoints2d(size=256)],
            "keypoints3d": [tasks.keypoints3d(size=256)],
            "edge_occlusion": [tasks.edge_occlusion(size=256)],
            "f(y^)": [tasks.reshading(size=256), tasks.principal_curvature(size=256)],
            "f(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.principal_curvature(size=256)],
            "s(y^)": [tasks.reshading(size=256), tasks.sobel_edges(size=256)],
            "s(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.sobel_edges(size=256)],
            "g(y^)": [tasks.reshading(size=256), tasks.depth_zbuffer(size=256)],
            "g(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.depth_zbuffer(size=256)],
            "nr(y^)": [tasks.reshading(size=256), tasks.normal(size=256)],
            "nr(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.normal(size=256)],
            "Nk2(y^)": [tasks.reshading(size=256), tasks.keypoints2d(size=256)],
            "Nk2(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.keypoints2d(size=256)],
            "Nk3(y^)": [tasks.reshading(size=256), tasks.keypoints3d(size=256)],
            "Nk3(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.keypoints3d(size=256)],
            "nEO(y^)": [tasks.reshading(size=256), tasks.edge_occlusion(size=256)],
            "nEO(n(x))": [tasks.rgb(size=256), tasks.reshading(size=256), tasks.edge_occlusion(size=256)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "normal",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_reshade_size320": {
        "paths": {
            "x": [tasks.rgb(size=320)],
            "y^": [tasks.reshading(size=320)],
            "n(x)": [tasks.rgb(size=320), tasks.reshading(size=320)],
            "RC(x)": [tasks.rgb(size=320), tasks.principal_curvature(size=320)],
            "a(x)": [tasks.rgb(size=320), tasks.sobel_edges(size=320)],
            "d(x)": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320)],
            "r(x)": [tasks.rgb(size=320), tasks.normal(size=320)],
            "k2(x)": [tasks.rgb(size=320), tasks.keypoints2d(size=320)],
            "k3(x)": [tasks.rgb(size=320), tasks.keypoints3d(size=320)],
            "EO(x)": [tasks.rgb(size=320), tasks.edge_occlusion(size=320)],
            "curv": [tasks.principal_curvature(size=320)],
            "edge": [tasks.sobel_edges(size=320)],
            "depth": [tasks.depth_zbuffer(size=320)],
            "reshading": [tasks.normal(size=320)],
            "keypoints2d": [tasks.keypoints2d(size=320)],
            "keypoints3d": [tasks.keypoints3d(size=320)],
            "edge_occlusion": [tasks.edge_occlusion(size=320)],
            "f(y^)": [tasks.reshading(size=320), tasks.principal_curvature(size=320)],
            "f(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.principal_curvature(size=320)],
            "s(y^)": [tasks.reshading(size=320), tasks.sobel_edges(size=320)],
            "s(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.sobel_edges(size=320)],
            "g(y^)": [tasks.reshading(size=320), tasks.depth_zbuffer(size=320)],
            "g(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.depth_zbuffer(size=320)],
            "nr(y^)": [tasks.reshading(size=320), tasks.normal(size=320)],
            "nr(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.normal(size=320)],
            "Nk2(y^)": [tasks.reshading(size=320), tasks.keypoints2d(size=320)],
            "Nk2(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.keypoints2d(size=320)],
            "Nk3(y^)": [tasks.reshading(size=320), tasks.keypoints3d(size=320)],
            "Nk3(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.keypoints3d(size=320)],
            "nEO(y^)": [tasks.reshading(size=320), tasks.edge_occlusion(size=320)],
            "nEO(n(x))": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.edge_occlusion(size=320)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_reshade_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.reshading(size=384)],
            "n(x)": [tasks.rgb(size=384), tasks.reshading(size=384)],
            "RC(x)": [tasks.rgb(size=384), tasks.principal_curvature(size=384)],
            "a(x)": [tasks.rgb(size=384), tasks.sobel_edges(size=384)],
            "d(x)": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384)],
            "r(x)": [tasks.rgb(size=384), tasks.normal(size=384)],
            "k2(x)": [tasks.rgb(size=384), tasks.keypoints2d(size=384)],
            "k3(x)": [tasks.rgb(size=384), tasks.keypoints3d(size=384)],
            "EO(x)": [tasks.rgb(size=384), tasks.edge_occlusion(size=384)],
            "curv": [tasks.principal_curvature(size=384)],
            "edge": [tasks.sobel_edges(size=384)],
            "depth": [tasks.depth_zbuffer(size=384)],
            "reshading": [tasks.normal(size=384)],
            "keypoints2d": [tasks.keypoints2d(size=384)],
            "keypoints3d": [tasks.keypoints3d(size=384)],
            "edge_occlusion": [tasks.edge_occlusion(size=384)],
            "f(y^)": [tasks.reshading(size=384), tasks.principal_curvature(size=384)],
            "f(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.principal_curvature(size=384)],
            "s(y^)": [tasks.reshading(size=384), tasks.sobel_edges(size=384)],
            "s(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.sobel_edges(size=384)],
            "g(y^)": [tasks.reshading(size=384), tasks.depth_zbuffer(size=384)],
            "g(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.depth_zbuffer(size=384)],
            "nr(y^)": [tasks.reshading(size=384), tasks.normal(size=384)],
            "nr(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.normal(size=384)],
            "Nk2(y^)": [tasks.reshading(size=384), tasks.keypoints2d(size=384)],
            "Nk2(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.keypoints2d(size=384)],
            "Nk3(y^)": [tasks.reshading(size=384), tasks.keypoints3d(size=384)],
            "Nk3(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.keypoints3d(size=384)],
            "nEO(y^)": [tasks.reshading(size=384), tasks.edge_occlusion(size=384)],
            "nEO(n(x))": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.edge_occlusion(size=384)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_reshade_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.reshading(size=448)],
            "n(x)": [tasks.rgb(size=448), tasks.reshading(size=448)],
            "RC(x)": [tasks.rgb(size=448), tasks.principal_curvature(size=448)],
            "a(x)": [tasks.rgb(size=448), tasks.sobel_edges(size=448)],
            "d(x)": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448)],
            "r(x)": [tasks.rgb(size=448), tasks.normal(size=448)],
            "k2(x)": [tasks.rgb(size=448), tasks.keypoints2d(size=448)],
            "k3(x)": [tasks.rgb(size=448), tasks.keypoints3d(size=448)],
            "EO(x)": [tasks.rgb(size=448), tasks.edge_occlusion(size=448)],
            "curv": [tasks.principal_curvature(size=448)],
            "edge": [tasks.sobel_edges(size=448)],
            "depth": [tasks.depth_zbuffer(size=448)],
            "reshading": [tasks.normal(size=448)],
            "keypoints2d": [tasks.keypoints2d(size=448)],
            "keypoints3d": [tasks.keypoints3d(size=448)],
            "edge_occlusion": [tasks.edge_occlusion(size=448)],
            "f(y^)": [tasks.reshading(size=448), tasks.principal_curvature(size=448)],
            "f(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.principal_curvature(size=448)],
            "s(y^)": [tasks.reshading(size=448), tasks.sobel_edges(size=448)],
            "s(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.sobel_edges(size=448)],
            "g(y^)": [tasks.reshading(size=448), tasks.depth_zbuffer(size=448)],
            "g(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.depth_zbuffer(size=448)],
            "nr(y^)": [tasks.reshading(size=448), tasks.normal(size=448)],
            "nr(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.normal(size=448)],
            "Nk2(y^)": [tasks.reshading(size=448), tasks.keypoints2d(size=448)],
            "Nk2(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.keypoints2d(size=448)],
            "Nk3(y^)": [tasks.reshading(size=448), tasks.keypoints3d(size=448)],
            "Nk3(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.keypoints3d(size=448)],
            "nEO(y^)": [tasks.reshading(size=448), tasks.edge_occlusion(size=448)],
            "nEO(n(x))": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.edge_occlusion(size=448)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=448, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_reshade_size512": {
        "paths": {
            "x": [tasks.rgb(size=512)],
            "y^": [tasks.reshading(size=512)],
            "n(x)": [tasks.rgb(size=512), tasks.reshading(size=512)],
            "RC(x)": [tasks.rgb(size=512), tasks.principal_curvature(size=512)],
            "a(x)": [tasks.rgb(size=512), tasks.sobel_edges(size=512)],
            "d(x)": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512)],
            "r(x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "k2(x)": [tasks.rgb(size=512), tasks.keypoints2d(size=512)],
            "k3(x)": [tasks.rgb(size=512), tasks.keypoints3d(size=512)],
            "EO(x)": [tasks.rgb(size=512), tasks.edge_occlusion(size=512)],
            "curv": [tasks.principal_curvature(size=512)],
            "edge": [tasks.sobel_edges(size=512)],
            "depth": [tasks.depth_zbuffer(size=512)],
            "reshading": [tasks.normal(size=512)],
            "keypoints2d": [tasks.keypoints2d(size=512)],
            "keypoints3d": [tasks.keypoints3d(size=512)],
            "edge_occlusion": [tasks.edge_occlusion(size=512)],
            "f(y^)": [tasks.reshading(size=512), tasks.principal_curvature(size=512)],
            "f(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.principal_curvature(size=512)],
            "s(y^)": [tasks.reshading(size=512), tasks.sobel_edges(size=512)],
            "s(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.sobel_edges(size=512)],
            "g(y^)": [tasks.reshading(size=512), tasks.depth_zbuffer(size=512)],
            "g(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.depth_zbuffer(size=512)],
            "nr(y^)": [tasks.reshading(size=512), tasks.normal(size=512)],
            "nr(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.normal(size=512)],
            "Nk2(y^)": [tasks.reshading(size=512), tasks.keypoints2d(size=512)],
            "Nk2(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.keypoints2d(size=512)],
            "Nk3(y^)": [tasks.reshading(size=512), tasks.keypoints3d(size=512)],
            "Nk3(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.keypoints3d(size=512)],
            "nEO(y^)": [tasks.reshading(size=512), tasks.edge_occlusion(size=512)],
            "nEO(n(x))": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.edge_occlusion(size=512)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=512, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_depth_size256": {
        "paths": {
            "x": [tasks.rgb(size=256)],
            "y^": [tasks.depth_zbuffer(size=256)],
            "n(x)": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256)],
            "RC(x)": [tasks.rgb(size=256), tasks.principal_curvature(size=256)],
            "a(x)": [tasks.rgb(size=256), tasks.sobel_edges(size=256)],
            "d(x)": [tasks.rgb(size=256), tasks.normal(size=256)],
            "r(x)": [tasks.rgb(size=256), tasks.reshading(size=256)],
            "k2(x)": [tasks.rgb(size=256), tasks.keypoints2d(size=256)],
            "k3(x)": [tasks.rgb(size=256), tasks.keypoints3d(size=256)],
            "EO(x)": [tasks.rgb(size=256), tasks.edge_occlusion(size=256)],
            "curv": [tasks.principal_curvature(size=256)],
            "edge": [tasks.sobel_edges(size=256)],
            "depth": [tasks.normal(size=256)],
            "reshading": [tasks.reshading(size=256)],
            "keypoints2d": [tasks.keypoints2d(size=256)],
            "keypoints3d": [tasks.keypoints3d(size=256)],
            "edge_occlusion": [tasks.edge_occlusion(size=256)],
            "f(y^)": [tasks.depth_zbuffer(size=256), tasks.principal_curvature(size=256)],
            "f(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.principal_curvature(size=256)],
            "s(y^)": [tasks.depth_zbuffer(size=256), tasks.sobel_edges(size=256)],
            "s(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.sobel_edges(size=256)],
            "g(y^)": [tasks.depth_zbuffer(size=256), tasks.normal(size=256)],
            "g(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.normal(size=256)],
            "nr(y^)": [tasks.depth_zbuffer(size=256), tasks.reshading(size=256)],
            "nr(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.reshading(size=256)],
            "Nk2(y^)": [tasks.depth_zbuffer(size=256), tasks.keypoints2d(size=256)],
            "Nk2(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.keypoints2d(size=256)],
            "Nk3(y^)": [tasks.depth_zbuffer(size=256), tasks.keypoints3d(size=256)],
            "Nk3(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.keypoints3d(size=256)],
            "nEO(y^)": [tasks.depth_zbuffer(size=256), tasks.edge_occlusion(size=256)],
            "nEO(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.edge_occlusion(size=256)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_depth_size256": {
        "paths": {
            "x": [tasks.rgb(size=256)],
            "y^": [tasks.depth_zbuffer(size=256)],
            "n(x)": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256)],
            "RC(x)": [tasks.rgb(size=256), tasks.principal_curvature(size=256)],
            "a(x)": [tasks.rgb(size=256), tasks.sobel_edges(size=256)],
            "d(x)": [tasks.rgb(size=256), tasks.normal(size=256)],
            "r(x)": [tasks.rgb(size=256), tasks.reshading(size=256)],
            "k2(x)": [tasks.rgb(size=256), tasks.keypoints2d(size=256)],
            "k3(x)": [tasks.rgb(size=256), tasks.keypoints3d(size=256)],
            "EO(x)": [tasks.rgb(size=256), tasks.edge_occlusion(size=256)],
            "curv": [tasks.principal_curvature(size=256)],
            "edge": [tasks.sobel_edges(size=256)],
            "depth": [tasks.normal(size=256)],
            "reshading": [tasks.reshading(size=256)],
            "keypoints2d": [tasks.keypoints2d(size=256)],
            "keypoints3d": [tasks.keypoints3d(size=256)],
            "edge_occlusion": [tasks.edge_occlusion(size=256)],
            "f(y^)": [tasks.depth_zbuffer(size=256), tasks.principal_curvature(size=256)],
            "f(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.principal_curvature(size=256)],
            "s(y^)": [tasks.depth_zbuffer(size=256), tasks.sobel_edges(size=256)],
            "s(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.sobel_edges(size=256)],
            "g(y^)": [tasks.depth_zbuffer(size=256), tasks.normal(size=256)],
            "g(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.normal(size=256)],
            "nr(y^)": [tasks.depth_zbuffer(size=256), tasks.reshading(size=256)],
            "nr(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.reshading(size=256)],
            "Nk2(y^)": [tasks.depth_zbuffer(size=256), tasks.keypoints2d(size=256)],
            "Nk2(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.keypoints2d(size=256)],
            "Nk3(y^)": [tasks.depth_zbuffer(size=256), tasks.keypoints3d(size=256)],
            "Nk3(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.keypoints3d(size=256)],
            "nEO(y^)": [tasks.depth_zbuffer(size=256), tasks.edge_occlusion(size=256)],
            "nEO(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.edge_occlusion(size=256)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_depth_size320": {
        "paths": {
            "x": [tasks.rgb(size=320)],
            "y^": [tasks.depth_zbuffer(size=320)],
            "n(x)": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320)],
            "RC(x)": [tasks.rgb(size=320), tasks.principal_curvature(size=320)],
            "a(x)": [tasks.rgb(size=320), tasks.sobel_edges(size=320)],
            "d(x)": [tasks.rgb(size=320), tasks.normal(size=320)],
            "r(x)": [tasks.rgb(size=320), tasks.reshading(size=320)],
            "k2(x)": [tasks.rgb(size=320), tasks.keypoints2d(size=320)],
            "k3(x)": [tasks.rgb(size=320), tasks.keypoints3d(size=320)],
            "EO(x)": [tasks.rgb(size=320), tasks.edge_occlusion(size=320)],
            "curv": [tasks.principal_curvature(size=320)],
            "edge": [tasks.sobel_edges(size=320)],
            "depth": [tasks.normal(size=320)],
            "reshading": [tasks.reshading(size=320)],
            "keypoints2d": [tasks.keypoints2d(size=320)],
            "keypoints3d": [tasks.keypoints3d(size=320)],
            "edge_occlusion": [tasks.edge_occlusion(size=320)],
            "f(y^)": [tasks.depth_zbuffer(size=320), tasks.principal_curvature(size=320)],
            "f(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.principal_curvature(size=320)],
            "s(y^)": [tasks.depth_zbuffer(size=320), tasks.sobel_edges(size=320)],
            "s(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.sobel_edges(size=320)],
            "g(y^)": [tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
            "g(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
            "nr(y^)": [tasks.depth_zbuffer(size=320), tasks.reshading(size=320)],
            "nr(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.reshading(size=320)],
            "Nk2(y^)": [tasks.depth_zbuffer(size=320), tasks.keypoints2d(size=320)],
            "Nk2(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.keypoints2d(size=320)],
            "Nk3(y^)": [tasks.depth_zbuffer(size=320), tasks.keypoints3d(size=320)],
            "Nk3(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.keypoints3d(size=320)],
            "nEO(y^)": [tasks.depth_zbuffer(size=320), tasks.edge_occlusion(size=320)],
            "nEO(n(x))": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.edge_occlusion(size=320)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_depth_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.depth_zbuffer(size=384)],
            "n(x)": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384)],
            "RC(x)": [tasks.rgb(size=384), tasks.principal_curvature(size=384)],
            "a(x)": [tasks.rgb(size=384), tasks.sobel_edges(size=384)],
            "d(x)": [tasks.rgb(size=384), tasks.normal(size=384)],
            "r(x)": [tasks.rgb(size=384), tasks.reshading(size=384)],
            "k2(x)": [tasks.rgb(size=384), tasks.keypoints2d(size=384)],
            "k3(x)": [tasks.rgb(size=384), tasks.keypoints3d(size=384)],
            "EO(x)": [tasks.rgb(size=384), tasks.edge_occlusion(size=384)],
            "curv": [tasks.principal_curvature(size=384)],
            "edge": [tasks.sobel_edges(size=384)],
            "depth": [tasks.normal(size=384)],
            "reshading": [tasks.reshading(size=384)],
            "keypoints2d": [tasks.keypoints2d(size=384)],
            "keypoints3d": [tasks.keypoints3d(size=384)],
            "edge_occlusion": [tasks.edge_occlusion(size=384)],
            "f(y^)": [tasks.depth_zbuffer(size=384), tasks.principal_curvature(size=384)],
            "f(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.principal_curvature(size=384)],
            "s(y^)": [tasks.depth_zbuffer(size=384), tasks.sobel_edges(size=384)],
            "s(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.sobel_edges(size=384)],
            "g(y^)": [tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
            "g(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
            "nr(y^)": [tasks.depth_zbuffer(size=384), tasks.reshading(size=384)],
            "nr(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.reshading(size=384)],
            "Nk2(y^)": [tasks.depth_zbuffer(size=384), tasks.keypoints2d(size=384)],
            "Nk2(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.keypoints2d(size=384)],
            "Nk3(y^)": [tasks.depth_zbuffer(size=384), tasks.keypoints3d(size=384)],
            "Nk3(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.keypoints3d(size=384)],
            "nEO(y^)": [tasks.depth_zbuffer(size=384), tasks.edge_occlusion(size=384)],
            "nEO(n(x))": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.edge_occlusion(size=384)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_depth_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.depth_zbuffer(size=448)],
            "n(x)": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448)],
            "RC(x)": [tasks.rgb(size=448), tasks.principal_curvature(size=448)],
            "a(x)": [tasks.rgb(size=448), tasks.sobel_edges(size=448)],
            "d(x)": [tasks.rgb(size=448), tasks.normal(size=448)],
            "r(x)": [tasks.rgb(size=448), tasks.reshading(size=448)],
            "k2(x)": [tasks.rgb(size=448), tasks.keypoints2d(size=448)],
            "k3(x)": [tasks.rgb(size=448), tasks.keypoints3d(size=448)],
            "EO(x)": [tasks.rgb(size=448), tasks.edge_occlusion(size=448)],
            "curv": [tasks.principal_curvature(size=448)],
            "edge": [tasks.sobel_edges(size=448)],
            "depth": [tasks.normal(size=448)],
            "reshading": [tasks.reshading(size=448)],
            "keypoints2d": [tasks.keypoints2d(size=448)],
            "keypoints3d": [tasks.keypoints3d(size=448)],
            "edge_occlusion": [tasks.edge_occlusion(size=448)],
            "f(y^)": [tasks.depth_zbuffer(size=448), tasks.principal_curvature(size=448)],
            "f(n(x))": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448), tasks.principal_curvature(size=448)],
            "s(y^)": [tasks.depth_zbuffer(size=448), tasks.sobel_edges(size=448)],
            "s(n(x))": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448), tasks.sobel_edges(size=448)],
            "g(y^)": [tasks.depth_zbuffer(size=448), tasks.normal(size=448)],
            "g(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.normal(size=256)],
            "nr(y^)": [tasks.depth_zbuffer(size=256), tasks.reshading(size=256)],
            "nr(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.reshading(size=256)],
            "Nk2(y^)": [tasks.depth_zbuffer(size=256), tasks.keypoints2d(size=256)],
            "Nk2(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.keypoints2d(size=256)],
            "Nk3(y^)": [tasks.depth_zbuffer(size=256), tasks.keypoints3d(size=256)],
            "Nk3(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.keypoints3d(size=256)],
            "nEO(y^)": [tasks.depth_zbuffer(size=256), tasks.edge_occlusion(size=256)],
            "nEO(n(x))": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256), tasks.edge_occlusion(size=256)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_size256": {
        "paths": {
            "x": [tasks.rgb(size=256)],
            "y^": [tasks.normal(size=256)],
            "n(x)": [tasks.rgb(size=256), tasks.normal(size=256)],
            "RC(x)": [tasks.rgb(size=256), tasks.principal_curvature(size=256)],
            "a(x)": [tasks.rgb(size=256), tasks.sobel_edges(size=256)],
            "d(x)": [tasks.rgb(size=256), tasks.depth_zbuffer(size=256)],
            "r(x)": [tasks.rgb(size=256), tasks.reshading(size=256)],
            "k2(x)": [tasks.rgb(size=256), tasks.keypoints2d(size=256)],
            "k3(x)": [tasks.rgb(size=256), tasks.keypoints3d(size=256)],
            "EO(x)": [tasks.rgb(size=256), tasks.edge_occlusion(size=256)],
            "curv": [tasks.principal_curvature(size=256)],
            "edge": [tasks.sobel_edges(size=256)],
            "depth": [tasks.depth_zbuffer(size=256)],
            "reshading": [tasks.reshading(size=256)],
            "keypoints2d": [tasks.keypoints2d(size=256)],
            "keypoints3d": [tasks.keypoints3d(size=256)],
            "edge_occlusion": [tasks.edge_occlusion(size=256)],
            "f(y^)": [tasks.normal(size=256), tasks.principal_curvature(size=256)],
            "f(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.principal_curvature(size=256)],
            "s(y^)": [tasks.normal(size=256), tasks.sobel_edges(size=256)],
            "s(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.sobel_edges(size=256)],
            "g(y^)": [tasks.normal(size=256), tasks.depth_zbuffer(size=256)],
            "g(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.depth_zbuffer(size=256)],
            "nr(y^)": [tasks.normal(size=256), tasks.reshading(size=256)],
            "nr(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.reshading(size=256)],
            "Nk2(y^)": [tasks.normal(size=256), tasks.keypoints2d(size=256)],
            "Nk2(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.keypoints2d(size=256)],
            "Nk3(y^)": [tasks.normal(size=256), tasks.keypoints3d(size=256)],
            "Nk3(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.keypoints3d(size=256)],
            "nEO(y^)": [tasks.normal(size=256), tasks.edge_occlusion(size=256)],
            "nEO(n(x))": [tasks.rgb(size=256), tasks.normal(size=256), tasks.edge_occlusion(size=256)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
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
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_size320": {
        "paths": {
            "x": [tasks.rgb(size=320)],
            "y^": [tasks.normal(size=320)],
            "n(x)": [tasks.rgb(size=320), tasks.normal(size=320)],
            "RC(x)": [tasks.rgb(size=320), tasks.principal_curvature(size=320)],
            "a(x)": [tasks.rgb(size=320), tasks.sobel_edges(size=320)],
            "d(x)": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320)],
            "r(x)": [tasks.rgb(size=320), tasks.reshading(size=320)],
            "k2(x)": [tasks.rgb(size=320), tasks.keypoints2d(size=320)],
            "k3(x)": [tasks.rgb(size=320), tasks.keypoints3d(size=320)],
            "EO(x)": [tasks.rgb(size=320), tasks.edge_occlusion(size=320)],
            "curv": [tasks.principal_curvature(size=320)],
            "edge": [tasks.sobel_edges(size=320)],
            "depth": [tasks.depth_zbuffer(size=320)],
            "reshading": [tasks.reshading(size=320)],
            "keypoints2d": [tasks.keypoints2d(size=320)],
            "keypoints3d": [tasks.keypoints3d(size=320)],
            "edge_occlusion": [tasks.edge_occlusion(size=320)],
            "f(y^)": [tasks.normal(size=320), tasks.principal_curvature(size=320)],
            "f(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.principal_curvature(size=320)],
            "s(y^)": [tasks.normal(size=320), tasks.sobel_edges(size=320)],
            "s(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.sobel_edges(size=320)],
            "g(y^)": [tasks.normal(size=320), tasks.depth_zbuffer(size=320)],
            "g(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.depth_zbuffer(size=320)],
            "nr(y^)": [tasks.normal(size=320), tasks.reshading(size=320)],
            "nr(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.reshading(size=320)],
            "Nk2(y^)": [tasks.normal(size=320), tasks.keypoints2d(size=320)],
            "Nk2(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.keypoints2d(size=320)],
            "Nk3(y^)": [tasks.normal(size=320), tasks.keypoints3d(size=320)],
            "Nk3(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.keypoints3d(size=320)],
            "nEO(y^)": [tasks.normal(size=320), tasks.edge_occlusion(size=320)],
            "nEO(n(x))": [tasks.rgb(size=320), tasks.normal(size=320), tasks.edge_occlusion(size=320)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=320, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_size384": {
        "paths": {
            "x": [tasks.rgb(size=384)],
            "y^": [tasks.normal(size=384)],
            "n(x)": [tasks.rgb(size=384), tasks.normal(size=384)],
            "RC(x)": [tasks.rgb(size=384), tasks.principal_curvature(size=384)],
            "a(x)": [tasks.rgb(size=384), tasks.sobel_edges(size=384)],
            "d(x)": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384)],
            "r(x)": [tasks.rgb(size=384), tasks.reshading(size=384)],
            "k2(x)": [tasks.rgb(size=384), tasks.keypoints2d(size=384)],
            "k3(x)": [tasks.rgb(size=384), tasks.keypoints3d(size=384)],
            "EO(x)": [tasks.rgb(size=384), tasks.edge_occlusion(size=384)],
            "curv": [tasks.principal_curvature(size=384)],
            "edge": [tasks.sobel_edges(size=384)],
            "depth": [tasks.depth_zbuffer(size=384)],
            "reshading": [tasks.reshading(size=384)],
            "keypoints2d": [tasks.keypoints2d(size=384)],
            "keypoints3d": [tasks.keypoints3d(size=384)],
            "edge_occlusion": [tasks.edge_occlusion(size=384)],
            "f(y^)": [tasks.normal(size=384), tasks.principal_curvature(size=384)],
            "f(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.principal_curvature(size=384)],
            "s(y^)": [tasks.normal(size=384), tasks.sobel_edges(size=384)],
            "s(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.sobel_edges(size=384)],
            "g(y^)": [tasks.normal(size=384), tasks.depth_zbuffer(size=384)],
            "g(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.depth_zbuffer(size=384)],
            "nr(y^)": [tasks.normal(size=384), tasks.reshading(size=384)],
            "nr(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.reshading(size=384)],
            "Nk2(y^)": [tasks.normal(size=384), tasks.keypoints2d(size=384)],
            "Nk2(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.keypoints2d(size=384)],
            "Nk3(y^)": [tasks.normal(size=384), tasks.keypoints3d(size=384)],
            "Nk3(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.keypoints3d(size=384)],
            "nEO(y^)": [tasks.normal(size=384), tasks.edge_occlusion(size=384)],
            "nEO(n(x))": [tasks.rgb(size=384), tasks.normal(size=384), tasks.edge_occlusion(size=384)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=384, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "baseline_size448": {
        "paths": {
            "x": [tasks.rgb(size=448)],
            "y^": [tasks.normal(size=448)],
            "n(x)": [tasks.rgb(size=448), tasks.normal(size=448)],
            "RC(x)": [tasks.rgb(size=448), tasks.principal_curvature(size=448)],
            "a(x)": [tasks.rgb(size=448), tasks.sobel_edges(size=448)],
            "d(x)": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448)],
            "r(x)": [tasks.rgb(size=448), tasks.reshading(size=448)],
            "k2(x)": [tasks.rgb(size=448), tasks.keypoints2d(size=448)],
            "k3(x)": [tasks.rgb(size=448), tasks.keypoints3d(size=448)],
            "EO(x)": [tasks.rgb(size=448), tasks.edge_occlusion(size=448)],
            "curv": [tasks.principal_curvature(size=448)],
            "edge": [tasks.sobel_edges(size=448)],
            "depth": [tasks.depth_zbuffer(size=448)],
            "reshading": [tasks.reshading(size=448)],
            "keypoints2d": [tasks.keypoints2d(size=448)],
            "keypoints3d": [tasks.keypoints3d(size=448)],
            "edge_occlusion": [tasks.edge_occlusion(size=448)],
            "f(y^)": [tasks.normal(size=448), tasks.principal_curvature(size=448)],
            "f(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.principal_curvature(size=448)],
            "s(y^)": [tasks.normal(size=448), tasks.sobel_edges(size=448)],
            "s(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.sobel_edges(size=448)],
            "g(y^)": [tasks.normal(size=448), tasks.depth_zbuffer(size=448)],
            "g(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.depth_zbuffer(size=448)],
            "nr(y^)": [tasks.normal(size=448), tasks.reshading(size=448)],
            "nr(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.reshading(size=448)],
            "Nk2(y^)": [tasks.normal(size=448), tasks.keypoints2d(size=448)],
            "Nk2(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.keypoints2d(size=448)],
            "Nk3(y^)": [tasks.normal(size=448), tasks.keypoints3d(size=448)],
            "Nk3(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.keypoints3d(size=448)],
            "nEO(y^)": [tasks.normal(size=448), tasks.edge_occlusion(size=448)],
            "nEO(n(x))": [tasks.rgb(size=448), tasks.normal(size=448), tasks.edge_occlusion(size=448)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=448, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },
    
    "baseline_size512": {
        "paths": {
            "x": [tasks.rgb(size=512)],
            "y^": [tasks.normal(size=512)],
            "n(x)": [tasks.rgb(size=512), tasks.normal(size=512)],
            "RC(x)": [tasks.rgb(size=512), tasks.principal_curvature(size=512)],
            "a(x)": [tasks.rgb(size=512), tasks.sobel_edges(size=512)],
            "d(x)": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512)],
            "r(x)": [tasks.rgb(size=512), tasks.reshading(size=512)],
            "k2(x)": [tasks.rgb(size=512), tasks.keypoints2d(size=512)],
            "k3(x)": [tasks.rgb(size=512), tasks.keypoints3d(size=512)],
            "EO(x)": [tasks.rgb(size=512), tasks.edge_occlusion(size=512)],
            "curv": [tasks.principal_curvature(size=512)],
            "edge": [tasks.sobel_edges(size=512)],
            "depth": [tasks.depth_zbuffer(size=512)],
            "reshading": [tasks.reshading(size=512)],
            "keypoints2d": [tasks.keypoints2d(size=512)],
            "keypoints3d": [tasks.keypoints3d(size=512)],
            "edge_occlusion": [tasks.edge_occlusion(size=512)],
            "f(y^)": [tasks.normal(size=512), tasks.principal_curvature(size=512)],
            "f(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.principal_curvature(size=512)],
            "s(y^)": [tasks.normal(size=512), tasks.sobel_edges(size=512)],
            "s(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.sobel_edges(size=512)],
            "g(y^)": [tasks.normal(size=512), tasks.depth_zbuffer(size=512)],
            "g(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.depth_zbuffer(size=512)],
            "nr(y^)": [tasks.normal(size=512), tasks.reshading(size=512)],
            "nr(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.reshading(size=512)],
            "Nk2(y^)": [tasks.normal(size=512), tasks.keypoints2d(size=512)],
            "Nk2(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.keypoints2d(size=512)],
            "Nk3(y^)": [tasks.normal(size=512), tasks.keypoints3d(size=512)],
            "Nk3(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.keypoints3d(size=512)],
            "nEO(y^)": [tasks.normal(size=512), tasks.edge_occlusion(size=512)],
            "nEO(n(x))": [tasks.rgb(size=512), tasks.normal(size=512), tasks.edge_occlusion(size=512)],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=512, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "curv",
                    "f(n(x))",
                    "edge",
                    "s(n(x))",
                    "depth",
                    "g(n(x))",
                    "reshading",
                    "nr(n(x))",
                    "keypoints2d",
                    "Nk2(n(x))",
                    "keypoints3d",
                    "Nk3(n(x))",
                    "edge_occlusion",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "percep_curv": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints": [tasks.keypoints3d],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk2(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
        },
        "freeze_list": [
            [tasks.normal, tasks.sobel_edges],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "indirect_curv": {
                ("train", "val"): [
                    ("f(n(x))", "curv"),
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
                    "s(y^)",
                    "s(n(x))",
                ]
            ),
        },
    },

    "multitask": {
        "paths": {
            "x": [tasks.rgb,],
            "normal_pred": [tasks.rgb, tasks.latent, tasks.normal],
            "curv_pred": [tasks.rgb, tasks.latent, tasks.principal_curvature],
            "depth_pred": [tasks.rgb, tasks.latent, tasks.depth_zbuffer],
            "reshade_pred": [tasks.rgb, tasks.latent, tasks.reshading],
            "keypoints_pred": [tasks.rgb, tasks.latent, tasks.keypoints3d],
            "occlusion_pred": [tasks.rgb, tasks.latent, tasks.edge_occlusion],
            "normal_target": [tasks.normal],
            "curv_target": [tasks.principal_curvature],
            "depth_target": [tasks.depth_zbuffer],
            "reshade_target": [tasks.reshading],
            "keypoints_target": [tasks.keypoints3d],
            "occlusion_target": [tasks.edge_occlusion],

            "normal_curv": [tasks.rgb, tasks.latent, tasks.normal, tasks.principal_curvature],
            "normal_depth": [tasks.rgb, tasks.latent, tasks.normal, tasks.depth_zbuffer],
            "normal_reshade": [tasks.rgb, tasks.latent, tasks.normal, tasks.reshading],
            "normal_keypoints": [tasks.rgb, tasks.latent, tasks.normal, tasks.keypoints3d],
            "normal_occlusion": [tasks.rgb, tasks.latent, tasks.normal, tasks.edge_occlusion],
            "curv_normal": [tasks.rgb, tasks.latent, tasks.principal_curvature, tasks.normal],
            "curv_depth": [tasks.rgb, tasks.latent, tasks.principal_curvature, tasks.depth_zbuffer],
            "curv_reshade": [tasks.rgb, tasks.latent, tasks.principal_curvature, tasks.reshading],
            "curv_keypoints": [tasks.rgb, tasks.latent, tasks.principal_curvature, tasks.keypoints3d],
            "curv_occlusion": [tasks.rgb, tasks.latent, tasks.principal_curvature, tasks.edge_occlusion],
            "depth_normal": [tasks.rgb, tasks.latent, tasks.depth_zbuffer, tasks.normal],
            "depth_curv": [tasks.rgb, tasks.latent, tasks.depth_zbuffer, tasks.principal_curvature],
            "depth_reshade": [tasks.rgb, tasks.latent, tasks.depth_zbuffer, tasks.reshading],
            "depth_keypoints": [tasks.rgb, tasks.latent, tasks.depth_zbuffer, tasks.keypoints3d],
            "depth_occlusion": [tasks.rgb, tasks.latent, tasks.depth_zbuffer, tasks.edge_occlusion],
            "reshade_normal": [tasks.rgb, tasks.latent, tasks.reshading, tasks.normal],
            "reshade_curv": [tasks.rgb, tasks.latent, tasks.reshading, tasks.principal_curvature],
            "reshade_depth": [tasks.rgb, tasks.latent, tasks.reshading, tasks.depth_zbuffer],
            "reshade_keypoints": [tasks.rgb, tasks.latent, tasks.reshading, tasks.keypoints3d],
            "reshade_occlusion": [tasks.rgb, tasks.latent, tasks.reshading, tasks.edge_occlusion],
            "keypoints_normal": [tasks.rgb, tasks.latent, tasks.keypoints3d, tasks.normal],
            "keypoints_curv": [tasks.rgb, tasks.latent, tasks.keypoints3d, tasks.principal_curvature],
            "keypoints_depth": [tasks.rgb, tasks.latent, tasks.keypoints3d, tasks.depth_zbuffer],
            "keypoints_reshade": [tasks.rgb, tasks.latent, tasks.keypoints3d, tasks.reshading],
            "keypoints_occlusion": [tasks.rgb, tasks.latent, tasks.keypoints3d, tasks.edge_occlusion],
            "occlusion_normal": [tasks.rgb, tasks.latent, tasks.edge_occlusion, tasks.normal],
            "occlusion_curv": [tasks.rgb, tasks.latent, tasks.edge_occlusion, tasks.principal_curvature],
            "occlusion_depth": [tasks.rgb, tasks.latent, tasks.edge_occlusion, tasks.depth_zbuffer],
            "occlusion_reshade": [tasks.rgb, tasks.latent, tasks.edge_occlusion, tasks.reshading],
            "occlusion_keypoints": [tasks.rgb, tasks.latent, tasks.edge_occlusion, tasks.keypoints3d],
        },
        "freeze_list": [
            # [tasks.normal, tasks.sobel_edges],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("normal_pred", "normal_target"),
                    ("curv_pred", "curv_target"),
                    ("depth_pred", "depth_target"),
                    ("reshade_pred", "reshade_target"),
                    ("keypoints_pred", "keypoints_target"),
                    ("occlusion_pred", "occlusion_target"),
                ],
            },
            # "energy": {
            #     ("train", "val"): [
            #         ("normal_curv", "normal_pred"),
            #         ("normal_depth", "normal_pred"),
            #         ("normal_reshade", "normal_pred"),
            #         ("normal_keypoints", "normal_pred"),
            #         ("normal_occlusion", "normal_pred"),
            #         ("curv_normal", "curv_pred"),
            #         ("curv_depth", "curv_pred"),
            #         ("curv_reshade", "curv_pred"),
            #         ("curv_keypoints", "curv_pred"),
            #         ("curv_occlusion", "curv_pred"),
            #         ("depth_normal", "depth_pred"),
            #         ("depth_curv", "depth_pred"),
            #         ("depth_reshade", "depth_pred"),
            #         ("depth_keypoints", "depth_pred"),
            #         ("depth_occlusion", "depth_pred"),
            #         ("reshade_normal", "reshade_pred"),
            #         ("reshade_curv", "reshade_pred"),
            #         ("reshade_depth", "reshade_pred"),
            #         ("reshade_keypoints", "reshade_pred"),
            #         ("reshade_occlusion", "reshade_pred"),
            #         ("keypoints_normal", "keypoints_pred"),
            #         ("keypoints_curv", "keypoints_pred"),
            #         ("keypoints_depth", "keypoints_pred"),
            #         ("keypoints_reshade", "keypoints_pred"),
            #         ("keypoints_occlusion", "keypoints_pred"),
            #         ("occlusion_normal", "occlusion_pred"),
            #         ("occlusion_curv", "occlusion_pred"),
            #         ("occlusion_depth", "occlusion_pred"),
            #         ("occlusion_reshade", "occlusion_pred"),
            #         ("occlusion_keypoints", "occlusion_pred"),
            #     ],
            # },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "normal_pred",
                    "normal_target",
                    "curv_pred",
                    "curv_target",
                    "depth_pred",
                    "depth_target",
                    "reshade_pred",
                    "reshade_target",
                    "keypoints_pred",
                    "keypoints_target",
                    "occlusion_pred",
                    "occlusion_target",
                ]
            ),
        },
    },


    "cycle": {
        "paths": {
            "x": [tasks.rgb,],
            "n(x)": [tasks.rgb, tasks.normal],
            "y^": [tasks.normal],
            "N(n(x))": [tasks.rgb, tasks.normal, tasks.rgb],
            "N(y^)": [tasks.normal, tasks.rgb],
        },
        "freeze_list": [
            [tasks.normal, tasks.rgb],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep": {
                ("train", "val"): [
                    ("N(n(x))", "x"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "n(x)",
                    "N(n(x))",
                    "N(y^)",
                ]
            ),
        },
    },

    "cycle_consistency": {
        "paths": {
            "x": [tasks.rgb,],
            "n(x)": [tasks.rgb, tasks.normal],
            "y^": [tasks.normal],
            "N(n(x))": [tasks.rgb, tasks.normal, tasks.rgb],
            "N(y^)": [tasks.normal, tasks.rgb],
        },
        "freeze_list": [
            [tasks.normal, tasks.rgb],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep": {
                ("train", "val"): [
                    ("N(n(x))", "N(y^)"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "n(x)",
                    "N(n(x))",
                    "N(y^)",
                ]
            ),
        },
    },

    "doublecycle": {
        "paths": {
            "x": [tasks.rgb,],
            "n(x)": [tasks.rgb, tasks.normal],
            "y^": [tasks.normal],
            "N(n(x))": [tasks.rgb, tasks.normal, tasks.rgb],
            "N(y^)": [tasks.normal, tasks.rgb],
        },
        "freeze_list": [
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep": {
                ("train", "val"): [
                    ("N(n(x))", "x"),
                    ("N(y^)", "x"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "x",
                    "n(x)",
                    "N(n(x))",
                    "N(y^)",
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
                    # IPython.embed()
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
                if loss_type != 'gan':
                    if loss_type not in loss:
                        loss[loss_type] = 0
                    for path1, path2 in losses:
                        
                        output_task = self.paths[path1][-1]
                        if "direct" in loss_type:
                            with torch.no_grad():
                                path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean)
                                loss[loss_type] += path_loss
                        else:
                            path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean)
                            loss[loss_type] += path_loss
                            self.metrics[reality.name][loss_type +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

                elif loss_type == 'gan':

                    for path1, path2 in losses:
                        if 'gan'+path1+path2 not in loss:
                            loss['disgan'+path1+path2] = 0
                            loss['graphgan'+path1+path2] = 0

                        ## Hack: detach() first pass so only OOD is updated
                        if path_values[path1] is None: continue
                        if path_values[path2] is None: continue
                        logit_path1 = discriminator(path_values[path1].detach())

                        ## Progressively increase GAN trade-off
                        #coeff = np.float(2.0 / (1.0 + np.exp(-10.0*self.train_iter / 10000.0)) - 1.0)
                        coeff = 0.1
                        path_value2 = path_values[path2] * 1.0
                        if reality.name == 'train':
                            path_value2.register_hook(coeff_hook(coeff))
                        logit_path2 = discriminator(path_value2)
                        binary_label = torch.Tensor([1]*logit_path1.size(0)+[0]*logit_path2.size(0)).float().cuda()
                        # print ("In BCE loss for gan: ", reality, logit_path1.mean(), logit_path2.mean())

                        gan_loss = nn.BCEWithLogitsLoss(size_average=True)(torch.cat((logit_path1,logit_path2), dim=0).view(-1), binary_label)
                        self.metrics[reality.name]['gan : ' + path1 + " -> " + path2] += [gan_loss.detach().cpu()]
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
                    name = loss_type +" : "+path1 + " -> " + path2
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

    def plot_paths_errors(self, graph, logger, realities=[], plot_names=None, prefix=""):
        

        error_pairs = {
            "n(x)": "y^",
            "f(n(x))": "curv",
            "s(n(x))": "edge",
            "g(n(x))": "depth",
            "nr(n(x))": "reshading",
            "Nk2(n(x))": "keypoints2d",
            "Nk3(n(x))": "keypoints3d",
            "nEO(n(x))": "edge_occlusion",
            "principal_curvature2normal": "y^",
            "sobel_edges2normal": "y^",
            "depth_zbuffer2normal": "y^",
            "reshading2normal": "y^",
            "edge_occlusion2normal": "y^",
            "keypoints3d2normal": "y^",
            "keypoints2d2normal": "y^",
            "rgb2normal": "normal",
            "rgb2principal_curvature": "principal_curvature",
            "rgb2sobel_edges": "sobel_edges",
            "rgb2depth_zbuffer": "depth_zbuffer",
            "rgb2reshading": "reshading",
            "rgb2edge_occlusion": "edge_occlusion",
            "rgb2keypoints3d": "keypoints3d",
            "rgb2keypoints2d": "keypoints2d",
        }

        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]
            realities = config["realities"]
            images = []

            cmap = get_cmap("jet")
            first = True

            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])
                shape = list(path_values[list(path_values.keys())[0]].shape)
                shape[1] = 3

                errors_list = []

                for i, path in enumerate(paths):
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    if first: images += [[]]
                    images[-1].append(X.clamp(min=0, max=1).expand(*shape))

                    if path in error_pairs:
                        if first: 
                            images += [[]]
                            images += [[]]

                        Y = path_values.get(path, torch.zeros(shape, device=DEVICE))
                        Y_hat = path_values.get(error_pairs[path], torch.zeros(shape, device=DEVICE))
                        out_task = self.paths[path][-1]

                        mask = ImageTask.build_mask(Y_hat, val=out_task.mask_val)
                        errors = ((Y - Y_hat)**2).mean(dim=1, keepdim=True)
                        log_errors = torch.log(errors.clamp(min=0, max=out_task.variance))

                        print (out_task, errors.max())
                        errors = (3*errors/(out_task.variance)).clamp(min=0, max=1)
                        log_errors = torch.log(errors)
                        log_errors = (log_errors - log_errors.min())/(log_errors.max() - log_errors.min())

                        errors = torch.tensor(cmap(errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        errors = errors.clamp(min=0, max=1).expand(*shape).to(DEVICE)
                        errors[~mask.expand_as(errors)] = 0.505
                        images[-2].append(errors)

                        errors_list.append(errors) 

                        log_errors = torch.tensor(cmap(log_errors.cpu()))[:, 0].permute((0, 3, 1, 2)).float()[:, 0:3]
                        log_errors = log_errors.clamp(min=0, max=1).expand(*shape).to(DEVICE)
                        log_errors[~mask.expand_as(log_errors)] = 0.505
                        images[-1].append(log_errors)

                    # mean_vals = np.mean(np.array(errors_list), axis=0)
                    # min_vals = np.amin(np.array(errors_list), axis=0)

                first = False

            for i in range(0, len(images)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]",
                resize=config["size"],
            )

    def plot_paths(self, graph, logger, realities=[], plot_names=None, prefix=""):
        
        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]
            realities = config["realities"]
            # images = None 
            images = [[] for _ in range(0, len(paths))]
            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])
                    # paths = path_values.keys()

                # if images is None:
                    # images = [[] for _ in range(0, len(paths))]
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






class NormalizedEnergyLoss(EnergyLoss):
    
    def __init__(self, *args, **kwargs):
        self.loss_weights = {}
        self.update_freq = kwargs.pop("update_freq", 30)
        self.iter = 0
        super().__init__(*args, **kwargs)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        if self.iter % self.update_freq == 0:
            self.loss_weights = {key: 1.0/loss.detach() for key, loss in loss_dict.items()}
        self.iter += 1
        return {key: loss * self.loss_weights[key] for key, loss in loss_dict.items()}





class CurriculumEnergyLoss(EnergyLoss):
    
    def __init__(self, *args, **kwargs):
        self.percep_weight = 0.0
        self.percep_step = kwargs.pop("percep_step", 0.1)
        super().__init__(*args, **kwargs)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        loss_dict["percep"] = loss_dict["percep"] * self.percep_weight
        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)
        self.percep_weight += self.percep_step
        logger.text (f'Current percep weight: {self.percep_weight}')




class WinRateEnergyLoss(EnergyLoss):
    
    def __init__(self, *args, **kwargs):
        
        self.k = kwargs.pop('k', 3)
        
        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[:7] == "percep_"]
        # last winrate computed on test set
        self.percep_winrate = {loss: 1.0 for loss in self.percep_losses}

        # running winrate stats
        self.running_stats = defaultdict(lambda: defaultdict(list))
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        
        loss_types = ["mse"] + \
                    [("percep_" + loss) for loss in self.chosen_losses] + \
                    [("direct_" + loss) for loss in self.chosen_losses] + \
                    [("indirect_" + loss) for loss in self.chosen_losses]
        
        loss_dict = super().__call__(graph, 
            discriminator=discriminator, 
            realities=realities, 
            loss_types=loss_types, 
            batch_mean=False
        )

        loss_dict["mse"] = loss_dict.pop("mse").mean()

        for key in self.chosen_losses:
            direct, indirect = loss_dict.pop(f"direct_{key}"), loss_dict.pop(f"indirect_{key}")
            
            # high winrate means direct is beating indirect significantly
            winrate = (direct < indirect).float()
            for reality in realities:
                self.running_stats[reality][key] += [list(winrate.detach().cpu().numpy())]
            
            loss_dict[f"percep_{key}"] = loss_dict.pop(f"percep_{key}").mean()

        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)

    def select_losses(self, reality):

        self.percep_winrate.update({loss: np.mean(value) for loss, value in self.running_stats[reality].items()})
        self.chosen_losses = sorted(
            self.percep_losses, 
            key=self.percep_winrate.get, 
            reverse=True
        ) [:self.k]
        self.running_stats[reality] = defaultdict(list)




# class WinRateEnergyLoss(EnergyLoss):
    
#     def __init__(self, *args, **kwargs):
#         self.k = kwargs.pop('k', 3)
#         self.random_select = kwargs.pop('random_select', False)
#         self.update_every_batch = kwargs.pop('update_every_batch', False)
#         self.percep_weight = kwargs.pop('percep_weight', 1.0)
#         self.percep_step = kwargs.pop('percep_step', 0.0)
#         self.standardize = kwargs.pop('standardize', False)
#         self.unit_mean = kwargs.pop('unit_mean', False)
#         self.running_stats = {}

#         super().__init__(*args, **kwargs)
#         self.select_losses()

#     def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        
#         if self.update_every_batch: self.select_losses()
#         loss_types = ["mse"] + \
#                     [("percep_" + loss) for loss in self.chosen_losses] + \
#                     [("direct_" + loss) for loss in self.chosen_losses]

#         loss_dict = super().__call__(graph, 
#             discriminator=discriminator, 
#             realities=realities, 
#             loss_types=loss_types, 
#             batch_mean=False
#         )

#         for key in self.chosen_losses:
#             percep, direct = loss_dict.pop(f"percep_{key}"), loss_dict.pop(f"direct_{key}")
#             winrate = torch.mean((percep > direct).float()) # winrate high means direct better than perceptual
#             self.running_stats[key] = winrate.detach().cpu().item()
#             if self.standardize:
#                 percep = torch.mean((percep - percep.mean())/percep.std())
#             if self.unit_mean:
#                 percep = percep.mean() / direct.detach().mean()
#             loss_dict[f"percep_{key}"] = percep * self.percep_weight
        
#         mse = loss_dict["mse"]
#         if self.standardize:
#             loss_dict["mse"] = torch.mean((mse - mse.mean())/mse.std())
#         if self.unit_mean:
#             loss_dict["mse"] = mse.mean() / mse.mean().detach()
        
#         return loss_dict

#     def logger_update(self, logger):
#         super().logger_update(logger)
#         self.select_losses()
#         logger.text (f"Chosen losses: {self.chosen_losses}")

#     def logger_hooks(self, logger):
#         if not self.update_every_batch:
#             super().logger_hooks(logger)
#             return

#         # Split up train and val because they are no longer necessarily equal 
#         name_to_realities = defaultdict(list)
#         for loss_type, loss_item in self.losses.items():
#             for realities, losses in loss_item.items():
#                 for path1, path2 in losses:
#                     name = loss_type+" : "+path1 + " -> " + path2
#                     name_to_realities[name] += list(realities)

#         for name, realities in name_to_realities.items():
#             for reality in realities:
#                 def plot(logger, data, name=name, reality=reality):
#                     logger.plot(data[f"{reality}_{name}"], f"{reality}_{name}")
#                 logger.add_hook(partial(plot, name=name, reality=reality), feature=f"{reality}_{name}", freq=1)


#     def select_losses(self):

#         self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
#         if self.random_select or len(self.running_stats) < len(self.percep_losses):
#             self.chosen_losses = random.sample(self.percep_losses, self.k)
#         else:
#             self.chosen_losses = sorted(self.running_stats, key=self.running_stats.get, reverse=True)[:self.k]
        

