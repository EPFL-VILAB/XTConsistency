import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph, Discriminator
from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask
from evaluation import run_eval_suite
from datasets import ImageDataset

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

from torchvision.utils import save_image

import IPython

def main(
    mode="standard", visualize=False,
    pretrained=True, finetuned=False, batch_size=None, 
    **kwargs,
):

    configs = {
        # "VISUALS2_rgb2normals2x_multipercep8_winrate_standardized_upd": dict(
        #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
        #     cont="mount/shared/results_LBP_multipercep8_winrate_standardized_upd_3/graph.pth",
        #     test=True, ood=True, oodfull=True,
        # ),
        # "VISUALS2_rgb2reshade2x_latwinrate_reshadetarget": dict(
        #     loss_configs=["baseline_reshade_size256", "baseline_reshade_size320", "baseline_reshade_size384", "baseline_reshade_size448", "baseline_reshade_size512"],
        #     cont="mount/shared/results_LBP_multipercep_latwinrate_reshadingtarget_6/graph.pth",
        #     test=True, ood=True, oodfull=True,
        # ),
        # "VISUALS2_rgb2reshade2x_reshadebaseline": dict(
        #     loss_configs=["baseline_reshade_size256", "baseline_reshade_size320", "baseline_reshade_size384", "baseline_reshade_size448", "baseline_reshade_size512"],
        #     test=True, ood=True, oodfull=True,
        # ),
        # "VISUALS2_rgb2reshade2x_latwinrate_depthtarget": dict(
        #     loss_configs=["baseline_depth_size256", "baseline_depth_size320", "baseline_depth_size384", "baseline_depth_size448", "baseline_depth_size512"],
        #     cont="mount/shared/results_LBP_multipercep_latwinrate_reshadingtarget_6/graph.pth",
        #     test=True, ood=True, oodfull=True,
        # ),
        # "VISUALS2_rgb2reshade2x_depthbaseline": dict(
        #     loss_configs=["baseline_depth_size256", "baseline_depth_size320", "baseline_depth_size384", "baseline_depth_size448", "baseline_depth_size512"],
        #     test=True, ood=True, oodfull=True,
        # ),
        # "VISUALS2_rgb2normals2x_baseline": dict(
        #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
        #     test=True, ood=True, oodfull=True,
        # ),
        # "VISUALS2_rgb2normals2x_multipercep": dict(
        #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
        #     test=True, ood=True, oodfull=True,
        #     cont="mount/shared/results_LBP_multipercep_32/graph.pth",
        # ),
        "VISUALS2_rgb2x2normals_baseline": dict(
            loss_configs=["rgb2x2normals_plots", "rgb2x2normals_plots_size320", "rgb2x2normals_plots_size384", "rgb2x2normals_plots_size448", "rgb2x2normals_plots_size512"],
            finetuned=False,
            test=True, ood=True, ood_full=True,
        ),
        "VISUALS2_rgb2x2normals_finetuned": dict(
            loss_configs=["rgb2x2normals_plots", "rgb2x_plots2normals_size320", "rgb2x2normals_plots_size384", "rgb2x2normals_plots_size448", "rgb2x2normals_plots_size512"],
            finetuned=True,
            test=True, ood=True, ood_full=True,
        ),
        "VISUALS2_rgb2x_baseline": dict(
            loss_configs=["rgb2x_plots", "rgb2x_plots_size320", "rgb2x_plots_size384", "rgb2x_plots_size448", "rgb2x_plots_size512"],
            finetuned=False,
            test=True, ood=True, ood_full=True,
        ),
        "VISUALS2_rgb2x_finetuned": dict(
            loss_configs=["rgb2x_plots", "rgb2x_plots_size320", "rgb2x_plots_size384", "rgb2x_plots_size448", "rgb2x_plots_size512"],
            finetuned=True,
            test=True, ood=True, ood_full=True,
        ),
    }

    # configs = {
    #   "VISUALS_rgb2normals2x_latv2": dict(
    #       loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
    #       cont="mount/shared/results_LBP_multipercep_latv2_10/graph.pth",
    #   ),
    #   "VISUALS_rgb2normals2x_lat_winrate": dict(
    #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
    #     cont="mount/shared/results_LBP_multipercep_lat_winrate_8/graph.pth",
    #   ),
    #   "VISUALS_rgb2normals2x_multipercep": dict(
    #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
    #     cont="mount/shared/results_LBP_multipercep_32/graph.pth",
    #   ),
    #   "VISUALS_rgb2normals2x_rndv2": dict(
    #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
    #     cont="mount/shared/results_LBP_multipercep_rnd_11/graph.pth",
    #   ),
    #   "VISUALS_rgb2normals2x_baseline": dict(
    #     loss_configs=["baseline_size256", "baseline_size320", "baseline_size384", "baseline_size448", "baseline_size512"],
    #     cont=None,
    #   ),
    #   "VISUALS_rgb2x2normals_baseline": dict(
    #     loss_configs=["rgb2x2normals_plots", "rgb2x2normals_plots_size320", "rgb2x2normals_plots_size384", "rgb2x2normals_plots_size448", "rgb2x2normals_plots_size512"],
    #     finetuned=False,
    #   ),
    #   "VISUALS_rgb2x2normals_finetuned": dict(
    #     loss_configs=["rgb2x2normals_plots", "rgb2x2normals_plots_size320", "rgb2x2normals_plots_size384", "rgb2x2normals_plots_size448", "rgb2x2normals_plots_size512"],
    #     finetuned=True,
    #   ),
    #   "VISUALS_y2normals_baseline": dict(
    #     loss_configs=["y2normals_plots", "y2normals_plots_size320", "y2normals_plots_size384", "y2normals_plots_size448", "y2normals_plots_size512"],
    #     finetuned=False,
    #   ),
    #   "VISUALS_y2normals_finetuned": dict(
    #     loss_configs=["y2normals_plots", "y2normals_plots_size320", "y2normals_plots_size384", "y2normals_plots_size448", "y2normals_plots_size512"],
    #     finetuned=True,
    #   ),
    #   "VISUALS_rgb2x_baseline": dict(
    #     loss_configs=["rgb2x_plots", "rgb2x_plots_size320", "rgb2x_plots_size384", "rgb2x_plots_size448", "rgb2x_plots_size512"],
    #     finetuned=False,
    #   ),
    #   "VISUALS_rgb2x_finetuned": dict(
    #     loss_configs=["rgb2x_plots", "rgb2x_plots_size320", "rgb2x_plots_size384", "rgb2x_plots_size448", "rgb2x_plots_size512"],
    #     finetuned=True,
    #   ),
    # }

    for i in range(0, 5):

        config = configs[list(configs.keys())[0]]

        finetuned = config.get("finetuned", False)
        loss_configs = config["loss_configs"]

        loss_config = loss_configs[i]

        batch_size = batch_size or 32
        energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

        # DATA LOADING 1
        test_set = load_test(energy_loss.get_tasks("test"), sample=8)

        ood_tasks = [task for task in energy_loss.get_tasks("ood") if task.kind == 'rgb']
        ood_set = load_ood(ood_tasks, sample=4)
        print (ood_tasks)
        
        test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
        ood = RealityTask.from_static("ood", ood_set, ood_tasks)

        # DATA LOADING 2
        ood_tasks = list(set([tasks.rgb] + [task for task in energy_loss.get_tasks("ood") if task.kind == 'rgb']))
        test_set = load_test(ood_tasks, sample=2)
        ood_set = load_ood(ood_tasks)

        test2 = RealityTask.from_static("test", test_set, ood_tasks)
        ood2 = RealityTask.from_static("ood", ood_set, ood_tasks)

        # DATA LOADING 3
        test_set = load_test(energy_loss.get_tasks("test"), sample=8)
        ood_tasks = [task for task in energy_loss.get_tasks("ood") if task.kind == 'rgb']

        ood_loader = torch.utils.data.DataLoader(
            ImageDataset(tasks=ood_tasks, data_dir=f"{SHARED_DIR}/ood_images"),
            batch_size=32,
            num_workers=32, shuffle=False, pin_memory=True
        )
        data = list(itertools.islice(ood_loader, 2))
        test_set = data[0]
        ood_set = data[1]
        
        test3 = RealityTask.from_static("test", test_set, ood_tasks)
        ood3 = RealityTask.from_static("ood", ood_set, ood_tasks)




        for name, config in configs.items():

            finetuned = config.get("finetuned", False)
            loss_configs = config["loss_configs"]
            cont = config.get("cont", None)

            logger = VisdomLogger("train", env=name, delete=True if i == 0 else False)
            if config.get("test", False):                
                # GRAPH
                realities = [test, ood]
                print ("Finetuned: ", finetuned)
                graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=finetuned, lazy=True)
                if cont is not None: graph.load_weights(cont)

                # LOGGING
                energy_loss.plot_paths_errors(graph, logger, realities, prefix=loss_config)

    
            logger = VisdomLogger("train", env=name + "_ood", delete=True if i == 0 else False)
            if config.get("ood", False):
                # GRAPH
                realities = [test2, ood2]
                print ("Finetuned: ", finetuned)
                graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=finetuned, lazy=True)
                if cont is not None: graph.load_weights(cont)

                energy_loss.plot_paths(graph, logger, realities, prefix=loss_config)

            logger = VisdomLogger("train", env=name + "_oodfull", delete=True if i == 0 else False)
            if config.get("oodfull", False):

                # GRAPH
                realities = [test3, ood3]
                print ("Finetuned: ", finetuned)
                graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=finetuned, lazy=True)
                if cont is not None: graph.load_weights(cont)

                energy_loss.plot_paths(graph, logger, realities, prefix=loss_config)


if __name__ == "__main__":
    Fire(main)
