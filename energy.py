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
    "consistency_paired_resolution_cycle_lowweight": {
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
            ("train", "val"): {
                ("n(x)", "y^"): 1.0,
                ("F(z^)", "y^"): 1.0,
                ("RC(x)", "z^"): 1.0,
                ("F(RC(x))", "y^"): 1.0,
                ("F(RC(x))", "n(x)"): 1.0,
                ("F(RC(~x))", "n(~x)"): 1.0,
                ("~n(~x)", "n(x)"): 0.05
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
    "consistency_paired_resolution_cycle_baseline_lowweight": {
        "paths": {
            "x": [tasks.rgb],
            "~x": [tasks.rgb(blur_radius=1)],
            "y^": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "F(z^)": [tasks.principal_curvature, tasks.normal],
            "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
            "n(~x)": [tasks.rgb(blur_radius=1), tasks.normal(blur_radius=1)],
            #"~n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3), tasks.normal],
            "F(RC(~x))": [tasks.rgb(blur_radius=1), tasks.principal_curvature(blur_radius=1), tasks.normal(blur_radius=1)],
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
            #"~n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3), tasks.normal],
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
                    ("F(RC(~x))", "n(~x)"),
                    #("~n(~x)", "n(x)"),
                ],
            },
            "gan": {
                ("train", "val"): [
                    ("n(x)", "n(~x)"),
                    ("F(RC(x))", "F(RC(~x))"),
                    ("y^", "n(~x)"),
                    ("y^", "F(RC(~x))"),
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
    # "rgb2x2normals_plots_size320": {
    #     "paths": {
    #         "x": [tasks.rgb(size=320)],
    #         "~x": [tasks.rgb],
    #         "~y": [tasks.normal],
    #         "y^": [tasks.normal(size=320)],
    #         "n(x)": [tasks.rgb(size=320), tasks.normal(size=320)],
    #         "principal_curvature": [tasks.rgb(size=320), tasks.principal_curvature(size=320), tasks.normal(size=320)],
    #         "sobel_edges": [tasks.rgb(size=320), tasks.sobel_edges(size=320), tasks.normal(size=320)],
    #         "depth_zbuffer": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
    #         "reshading": [tasks.rgb(size=320), tasks.reshading(size=320), tasks.normal(size=320)],
    #         "edge_occlusion": [tasks.rgb(size=320), tasks.edge_occlusion(size=320), tasks.normal(size=320)],
    #         "keypoints3d": [tasks.rgb(size=320), tasks.keypoints3d(size=320), tasks.normal(size=320)],
    #         "keypoints2d": [tasks.rgb(size=320), tasks.keypoints2d(size=320), tasks.normal(size=320)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("n(x)", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=320, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x2normals_plots_size384": {
    #     "paths": {
    #         "x": [tasks.rgb(size=384)],
    #         "y^": [tasks.normal(size=384)],
    #         "n(x)": [tasks.rgb(size=384), tasks.normal(size=384)],
    #         "principal_curvature": [tasks.rgb(size=384), tasks.principal_curvature(size=384), tasks.normal(size=384)],
    #         "sobel_edges": [tasks.rgb(size=384), tasks.sobel_edges(size=384), tasks.normal(size=384)],
    #         "depth_zbuffer": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
    #         "reshading": [tasks.rgb(size=384), tasks.reshading(size=384), tasks.normal(size=384)],
    #         "edge_occlusion": [tasks.rgb(size=384), tasks.edge_occlusion(size=384), tasks.normal(size=384)],
    #         "keypoints3d": [tasks.rgb(size=384), tasks.keypoints3d(size=384), tasks.normal(size=384)],
    #         "keypoints2d": [tasks.rgb(size=384), tasks.keypoints2d(size=384), tasks.normal(size=384)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("n(x)", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=384, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x2normals_plots_size448": {
    #     "paths": {
    #         "x": [tasks.rgb(size=448)],
    #         "y^": [tasks.normal(size=448)],
    #         "n(x)": [tasks.rgb(size=448), tasks.normal(size=448)],
    #         "principal_curvature": [tasks.rgb(size=448), tasks.principal_curvature(size=448), tasks.normal(size=448)],
    #         "sobel_edges": [tasks.rgb(size=448), tasks.sobel_edges(size=448), tasks.normal(size=448)],
    #         "depth_zbuffer": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448), tasks.normal(size=448)],
    #         "reshading": [tasks.rgb(size=448), tasks.reshading(size=448), tasks.normal(size=448)],
    #         "edge_occlusion": [tasks.rgb(size=448), tasks.edge_occlusion(size=448), tasks.normal(size=448)],
    #         "keypoints3d": [tasks.rgb(size=448), tasks.keypoints3d(size=448), tasks.normal(size=448)],
    #         "keypoints2d": [tasks.rgb(size=448), tasks.keypoints2d(size=448), tasks.normal(size=448)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("n(x)", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=448, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x2normals_plots_size512": {
    #     "paths": {
    #         "x": [tasks.rgb(size=512)],
    #         "y^": [tasks.normal(size=512)],
    #         "n(x)": [tasks.rgb(size=512), tasks.normal(size=512)],
    #         "principal_curvature": [tasks.rgb(size=512), tasks.principal_curvature(size=512), tasks.normal(size=512)],
    #         "sobel_edges": [tasks.rgb(size=512), tasks.sobel_edges(size=512), tasks.normal(size=512)],
    #         "depth_zbuffer": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512), tasks.normal(size=512)],
    #         "reshading": [tasks.rgb(size=512), tasks.reshading(size=512), tasks.normal(size=512)],
    #         "edge_occlusion": [tasks.rgb(size=512), tasks.edge_occlusion(size=512), tasks.normal(size=512)],
    #         "keypoints3d": [tasks.rgb(size=512), tasks.keypoints3d(size=512), tasks.normal(size=512)],
    #         "keypoints2d": [tasks.rgb(size=512), tasks.keypoints2d(size=512), tasks.normal(size=512)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("n(x)", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=512, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "y2normals_plots": {
    #     "paths": {
    #         "x": [tasks.rgb],
    #         "y^": [tasks.normal],
    #         "principal_curvature": [tasks.principal_curvature, tasks.normal],
    #         "sobel_edges": [tasks.sobel_edges, tasks.normal],
    #         "depth_zbuffer": [tasks.depth_zbuffer, tasks.normal],
    #         "reshading": [tasks.reshading, tasks.normal],
    #         "edge_occlusion": [tasks.edge_occlusion, tasks.normal],
    #         "keypoints3d": [tasks.keypoints3d, tasks.normal],
    #         "keypoints2d": [tasks.keypoints2d, tasks.normal],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=256, 
    #             realities=("test",), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "y2normals_plots_size320": {
    #     "paths": {
    #         "x": [tasks.rgb(size=320)],
    #         "y^": [tasks.normal(size=320)],
    #         "principal_curvature": [tasks.principal_curvature(size=320), tasks.normal(size=320)],
    #         "sobel_edges": [tasks.sobel_edges(size=320), tasks.normal(size=320)],
    #         "depth_zbuffer": [tasks.depth_zbuffer(size=320), tasks.normal(size=320)],
    #         "reshading": [tasks.reshading(size=320), tasks.normal(size=320)],
    #         "edge_occlusion": [tasks.edge_occlusion(size=320), tasks.normal(size=320)],
    #         "keypoints3d": [tasks.keypoints3d(size=320), tasks.normal(size=320)],
    #         "keypoints2d": [tasks.keypoints2d(size=320), tasks.normal(size=320)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=320, 
    #             realities=("test",), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "y2normals_plots_size384": {
    #     "paths": {
    #         "x": [tasks.rgb(size=384)],
    #         "y^": [tasks.normal(size=384)],
    #         "principal_curvature": [tasks.principal_curvature(size=384), tasks.normal(size=384)],
    #         "sobel_edges": [tasks.sobel_edges(size=384), tasks.normal(size=384)],
    #         "depth_zbuffer": [tasks.depth_zbuffer(size=384), tasks.normal(size=384)],
    #         "reshading": [tasks.reshading(size=384), tasks.normal(size=384)],
    #         "edge_occlusion": [tasks.edge_occlusion(size=384), tasks.normal(size=384)],
    #         "keypoints3d": [tasks.keypoints3d(size=384), tasks.normal(size=384)],
    #         "keypoints2d": [tasks.keypoints2d(size=384), tasks.normal(size=384)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=384, 
    #             realities=("test",), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "y2normals_plots_size448": {
    #     "paths": {
    #         "x": [tasks.rgb(size=448)],
    #         "y^": [tasks.normal(size=448)],
    #         "principal_curvature": [tasks.principal_curvature(size=448), tasks.normal(size=448)],
    #         "sobel_edges": [tasks.sobel_edges(size=448), tasks.normal(size=448)],
    #         "depth_zbuffer": [tasks.depth_zbuffer(size=448), tasks.normal(size=448)],
    #         "reshading": [tasks.reshading(size=448), tasks.normal(size=448)],
    #         "edge_occlusion": [tasks.edge_occlusion(size=448), tasks.normal(size=448)],
    #         "keypoints3d": [tasks.keypoints3d(size=448), tasks.normal(size=448)],
    #         "keypoints2d": [tasks.keypoints2d(size=448), tasks.normal(size=448)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=448, 
    #             realities=("test",), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "y2normals_plots_size512": {
    #     "paths": {
    #         "x": [tasks.rgb(size=512)],
    #         "y^": [tasks.normal(size=512)],
    #         "principal_curvature": [tasks.principal_curvature(size=512), tasks.normal(size=512)],
    #         "sobel_edges": [tasks.sobel_edges(size=512), tasks.normal(size=512)],
    #         "depth_zbuffer": [tasks.depth_zbuffer(size=512), tasks.normal(size=512)],
    #         "reshading": [tasks.reshading(size=512), tasks.normal(size=512)],
    #         "edge_occlusion": [tasks.edge_occlusion(size=512), tasks.normal(size=512)],
    #         "keypoints3d": [tasks.keypoints3d(size=512), tasks.normal(size=512)],
    #         "keypoints2d": [tasks.keypoints2d(size=512), tasks.normal(size=512)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "y^"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=512, 
    #             realities=("test",), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x_plots": {
    #     "paths": {
    #         "x": [tasks.rgb],
    #         "normal": [tasks.rgb, tasks.normal],
    #         "principal_curvature": [tasks.rgb, tasks.principal_curvature],
    #         "sobel_edges": [tasks.rgb, tasks.sobel_edges],
    #         "depth_zbuffer": [tasks.rgb, tasks.depth_zbuffer],
    #         "reshading": [tasks.rgb, tasks.reshading],
    #         "edge_occlusion": [tasks.rgb, tasks.edge_occlusion],
    #         "keypoints3d": [tasks.rgb, tasks.keypoints3d],
    #         "keypoints2d": [tasks.rgb, tasks.keypoints2d],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "normal"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=256, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "normal",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x_plots_size320": {
    #     "paths": {
    #         "x": [tasks.rgb(size=320)],
    #         "normal": [tasks.rgb(size=320), tasks.normal(size=320)],
    #         "principal_curvature": [tasks.rgb(size=320), tasks.principal_curvature(size=320)],
    #         "sobel_edges": [tasks.rgb(size=320), tasks.sobel_edges(size=320)],
    #         "depth_zbuffer": [tasks.rgb(size=320), tasks.depth_zbuffer(size=320)],
    #         "reshading": [tasks.rgb(size=320), tasks.reshading(size=320)],
    #         "edge_occlusion": [tasks.rgb(size=320), tasks.edge_occlusion(size=320)],
    #         "keypoints3d": [tasks.rgb(size=320), tasks.keypoints3d(size=320)],
    #         "keypoints2d": [tasks.rgb(size=320), tasks.keypoints2d(size=320)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "normal"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=320, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "normal",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x_plots_size384": {
    #     "paths": {
    #         "x": [tasks.rgb(size=384)],
    #         "normal": [tasks.rgb(size=384), tasks.normal(size=384)],
    #         "principal_curvature": [tasks.rgb(size=384), tasks.principal_curvature(size=384)],
    #         "sobel_edges": [tasks.rgb(size=384), tasks.sobel_edges(size=384)],
    #         "depth_zbuffer": [tasks.rgb(size=384), tasks.depth_zbuffer(size=384)],
    #         "reshading": [tasks.rgb(size=384), tasks.reshading(size=384)],
    #         "edge_occlusion": [tasks.rgb(size=384), tasks.edge_occlusion(size=384)],
    #         "keypoints3d": [tasks.rgb(size=384), tasks.keypoints3d(size=384)],
    #         "keypoints2d": [tasks.rgb(size=384), tasks.keypoints2d(size=384)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "normal"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=384, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "normal",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x_plots_size448": {
    #     "paths": {
    #         "x": [tasks.rgb(size=448)],
    #         "normal": [tasks.rgb(size=448), tasks.normal(size=448)],
    #         "principal_curvature": [tasks.rgb(size=448), tasks.principal_curvature(size=448)],
    #         "sobel_edges": [tasks.rgb(size=448), tasks.sobel_edges(size=448)],
    #         "depth_zbuffer": [tasks.rgb(size=448), tasks.depth_zbuffer(size=448)],
    #         "reshading": [tasks.rgb(size=448), tasks.reshading(size=448)],
    #         "edge_occlusion": [tasks.rgb(size=448), tasks.edge_occlusion(size=448)],
    #         "keypoints3d": [tasks.rgb(size=448), tasks.keypoints3d(size=448)],
    #         "keypoints2d": [tasks.rgb(size=448), tasks.keypoints2d(size=448)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "normal"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=448, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "normal",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "rgb2x_plots_size512": {
    #     "paths": {
    #         "x": [tasks.rgb(size=512)],
    #         "normal": [tasks.rgb(size=512), tasks.normal(size=512)],
    #         "principal_curvature": [tasks.rgb(size=512), tasks.principal_curvature(size=512)],
    #         "sobel_edges": [tasks.rgb(size=512), tasks.sobel_edges(size=512)],
    #         "depth_zbuffer": [tasks.rgb(size=512), tasks.depth_zbuffer(size=512)],
    #         "reshading": [tasks.rgb(size=512), tasks.reshading(size=512)],
    #         "edge_occlusion": [tasks.rgb(size=512), tasks.edge_occlusion(size=512)],
    #         "keypoints3d": [tasks.rgb(size=512), tasks.keypoints3d(size=512)],
    #         "keypoints2d": [tasks.rgb(size=512), tasks.keypoints2d(size=512)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("x", "normal"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=512, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "normal",
    #                 "principal_curvature",
    #                 "sobel_edges",
    #                 "depth_zbuffer",
    #                 "reshading",
    #                 "edge_occlusion",
    #                 "keypoints3d",
    #                 "keypoints2d",
    #             ]
    #         ),
    #     },
    # },
    # "consistency_paired_gaussianblur": {
    #     "paths": {
    #         "x": [tasks.rgb],
    #         "~x": [tasks.rgb(blur_radius=3)],
    #         "y^": [tasks.normal],
    #         "z^": [tasks.principal_curvature],
    #         "n(x)": [tasks.rgb, tasks.normal],
    #         "RC(x)": [tasks.rgb, tasks.principal_curvature],
    #         "F(z^)": [tasks.principal_curvature, tasks.normal],
    #         "F(RC(x))": [tasks.rgb, tasks.principal_curvature, tasks.normal],
    #         "n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3)],
    #         "~n(~x)": [tasks.rgb(blur_radius=3), tasks.normal(blur_radius=3), tasks.normal],
    #         "F(RC(~x))": [tasks.rgb(blur_radius=3), tasks.principal_curvature(blur_radius=3), tasks.normal(blur_radius=3)],
    #     },
    #     "losses": {
    #         ("train", "val"): [
    #             ("n(x)", "y^"),
    #             ("F(z^)", "y^"),
    #             ("RC(x)", "z^"),
    #             ("F(RC(x))", "y^"),
    #             ("F(RC(x))", "n(x)"),
    #             ("F(RC(~x))", "n(~x)"),
    #             ("~n(~x)", "n(x)"),
    #         ],
    #     },
    #     "plots": {
    #         "ID": dict(
    #             size=256, 
    #             realities=("test", "ood"), 
    #             paths=[
    #                 "x",
    #                 "y^",
    #                 "n(x)",
    #                 "F(RC(x))",
    #                 "z^",
    #                 "RC(x)",
    #             ]
    #         ),
    #         "OOD": dict(
    #             size=512, 
    #             realities=("test", "ood"),
    #             paths=[
    #                 "~x",
    #                 "n(~x)",
    #                 "F(RC(~x))",
    #             ]
    #         ),
    #     },
    # },
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

    def __call__(self, graph, discriminator_dict=None, realities=[]):
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
                            loss['gan'+path1+path2] = 0
                        logit_path1 = discriminator_dict[path1+path2](path_values[path1])
                        logit_path2 = discriminator_dict[path1+path2](path_values[path2])
                        binary_label = torch.Tensor([1]*logit_path1.size(0)+[0]*logit_path2.size(0)).float().cuda()
                        gan_loss = nn.BCEWithLogitsLoss()(torch.cat((logit_path1,logit_path2), dim=0).view(-1), binary_label)
                        self.metrics[reality.name]['gan : '+path1 + " -> " + path2] += [gan_loss.detach().cpu()]
                        loss['gan'+path1+path2] -= gan_loss 
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