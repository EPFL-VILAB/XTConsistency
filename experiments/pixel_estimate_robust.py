
import os, sys, math, random, itertools, pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from models import TrainableModel, DataParallelModel, WrapperModel
from logger import Logger, VisdomLogger
from task_configs import get_task, tasks, RealityTask
from transfers import functional_transfers
from datasets import TaskDataset
from graph import TaskGraph
from transfers import TRANSFER_MAP
from fire import Fire
import IPython


def main():

    task_list = [
        tasks.rgb, 
        tasks.normal, 
        tasks.principal_curvature, 
        tasks.sobel_edges,
        tasks.depth_zbuffer,
        tasks.reshading,
        tasks.edge_occlusion,
        tasks.keypoints3d,
        # tasks.keypoints2d,
    ]

    logger = VisdomLogger("train", env=JOB)

    reality = RealityTask('almena', 
        dataset=TaskDataset(buildings=['almena'], 
            tasks=task_list,
        ),
        tasks=task_list,
        shuffle=False,
        batch_size=4
    )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        anchored_tasks=[reality, tasks.rgb],
        reality=reality,
        batch_size=4,
        edges_exclude=[
            ('almena', 'normal'),
            ('almena', 'principal_curvature'),
            ('almena', 'sobel_edges'),
            ('almena', 'depth_zbuffer'),
            ('almena', 'reshading'),
            ('almena', 'keypoints2d'),
        ],
        initialize_first_order=True,
    )
    # assume all estimates are rgb2x
    # Show that you can get a gain with the appropriate neigbor weighting (either GT or rgb2y)
    # quantify this
    
    for task in [tasks.normal, tasks.principal_curvature, tasks.depth_zbuffer]:
        Y = graph.estimate(task)
        Y_hat = reality.task_data[task].to(DEVICE).detach()
        mse_orig = task.norm(Y, Y_hat)[0].data.cpu().numpy().mean()

        in_neighbors = [transfer for transfer in graph.in_adj[task] \
            if not isinstance(transfer, RealityTask) and transfer.src_task is not tasks.rgb]

        estimates = [
            transfer(graph.estimate(transfer.src_task)).detach() \
            for transfer in in_neighbors
        ]
        estimates = torch.stack(estimates, dim=1)
        mean = torch.mean(estimates, dim=1)
        median, _ = torch.median(estimates, dim=1)
        std = torch.std(estimates, dim=1)

        sorted_arr, _ = torch.sort(estimates, dim=1)
        robust_mean = torch.mean(sorted_arr[:, 2:-2], dim=1)
        
        update = Y + ((mean - Y)/(std)).clamp(min=-3.0, max=3.0).abs()/3.0 * (robust_mean - Y)
        update_random = Y + ((mean - Y)/(std.mean())).clamp(min=-3.0, max=3.0).abs()/3.0 * (robust_mean - Y)

        print (Y.shape, mean.shape, median.shape, robust_mean.shape, update.shape, Y_hat.shape)
        logger.images_grouped([Y, update, update_random, mean, median, robust_mean, Y_hat], task.name, resize=192)


if __name__ == "__main__":
    Fire(main)


