
import os, sys, math, random, itertools, pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from task_configs import get_task, tasks, RealityTask
from transfers import functional_transfers
from datasets import TaskDataset, SintelDataset
from graph import TaskGraph

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
        tasks.keypoints2d,
    ]
    # tasks.depth_zbuffer.image_transform = tasks.depth_zbuffer.sintel_depth.image_transform

    reality = RealityTask('sintel', 
        dataset=SintelDataset(tasks=[tasks.rgb, tasks.normal]),
        tasks=[tasks.rgb, tasks.normal],
        batch_size=4
    )
    graph = TaskGraph(
        tasks=[reality, *task_list],
        anchored_tasks=[reality, tasks.rgb, tasks.normal],
        reality=reality,
        batch_size=4,
        edges_exclude=[
            #('sintel', 'normal'),
        ],
        initialize_first_order=True,
    )

    graph.p.compile(torch.optim.Adam, lr=4e-2)
    graph.estimates.compile(torch.optim.Adam, lr=1e-2)

    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="energy", freq=16)
    logger.add_hook(lambda logger, data: logger.plot(data["energy"], "free_energy"), feature="energy", freq=100)
    logger.add_hook(lambda logger, data: graph.plot_estimates(logger), feature="epoch", freq=32)
    # logger.add_hook(lambda logger, data: graph.update_paths(logger), feature="epoch", freq=32)

    # graph.plot_estimates(logger)
    # graph.plot_paths(logger, dest_tasks=[tasks.normal], show_images=True)
    # graph.update_paths(logger)

    for epochs in range(0, 4000):
        logger.update("epoch", epochs)
        free_energy = graph.free_energy(sample=8)
        graph.estimates.step(free_energy)
        logger.update("energy", free_energy)


if __name__ == "__main__":
    Fire(main)


