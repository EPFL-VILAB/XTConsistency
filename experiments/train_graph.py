
import os, sys, math, random, itertools, pickle, heapq
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
    ]

    reality = RealityTask('almena', 
        dataset=TaskDataset(
            buildings=['almena'],
            tasks=task_list,
        ),
        tasks=task_list,
        batch_size=4
    )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        edges_exclude=[
            ('almena', 'normal'),  #remove all GT for normals
            # ('almena', 'principal_curvature'),  #remove all GT for normals
            ('almena', 'depth_zbuffer'),  #remove all GT for normals
            ('almena', 'sobel_edges'),  #remove all GT for normals
            # ('rgb', 'normal'), #remove all GT substitutes (pretraining?)
            # ('rgb', 'principal_curvature'), #remove all GT substitutes (pretraining?)
            # ('rgb', 'depth_zbuffer'), #remove all GT substitutes (pretraining?)
            # ('rgb', 'sobel_edges'), #remove all GT substitutes (pretraining?)
        ],
        anchored_tasks=[
            reality,
            tasks.rgb,
            tasks.principal_curvature,
            # tasks.depth_zbuffer,
            # tasks.sobel_edges,
        ],
        batch_size=4
    )

    print (graph.edges)
    graph.p.compile(torch.optim.Adam, lr=4e-2)
    graph.estimates.compile(torch.optim.Adam, lr=1e-2)
    graph.estimates['rgb'].data = reality.task_data[tasks.rgb].to(DEVICE)
    graph.estimates['principal_curvature'].data = reality.task_data[tasks.principal_curvature].to(DEVICE)
    # graph.estimates['normal'].data = TRANSFER_MAP['n'](graph.estimates['rgb'].data).detach().to(DEVICE)

    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="energy", freq=16)
    logger.add_hook(lambda logger, data: logger.plot(data["energy"], "free_energy"), feature="energy", freq=100)
    logger.add_hook(lambda logger, data: graph.plot_estimates(logger), feature="epoch", freq=32)
    logger.add_hook(lambda logger, data: graph.update_paths(logger, reality), feature="epoch", freq=32)

    graph.plot_paths(logger, reality, show_images=True)

    for epochs in range(0, 4000):
        logger.update("epoch", epochs)

        free_energy = graph.free_energy(sample=12)
        graph.estimates.step(free_energy)
        logger.update("energy", free_energy)


if __name__ == "__main__":
    Fire(main)

