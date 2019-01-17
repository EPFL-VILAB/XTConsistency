
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
from datasets import TaskDataset
from graph import TaskGraph, ProbabilisticTaskGraph
from transfers import TRANSFER_MAP
from fire import Fire
import IPython


def main(batch_size=2):

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

    reality = RealityTask('almena', 
        dataset=TaskDataset(buildings=['almena'], 
            tasks=task_list,
        ),
        tasks=task_list,
        batch_size=batch_size
    )

    graph = ProbabilisticTaskGraph(
        tasks=[reality, *task_list],
        anchored_tasks=[
            tasks.rgb,
            tasks.principal_curvature,
            tasks.sobel_edges,
            tasks.depth_zbuffer,
            tasks.reshading,
            tasks.edge_occlusion,
            tasks.keypoints3d,
            tasks.keypoints2d,
        ],
        reality=reality,
        batch_size=batch_size,
        edges_exclude=[
            ('almena', 'normal'),
        ],
        initialize_first_order=False,
    )
    graph.estimates["normal"].data = torch.randn_like(graph.estimate(tasks.normal))*tasks.normal.variance
    print (graph.edges)

    graph.compile(torch.optim.Adam, lr=5e-2)

    logger = VisdomLogger("train", env=JOB)
    # logger.add_hook(lambda logger, data: logger.step(), feature="energy", freq=16)
    logger.add_hook(lambda logger, data: logger.plot(data["energy"], "free_energy"), feature="energy", freq=32)
    logger.add_hook(lambda logger, data: graph.plot_estimates(logger), feature="epoch", freq=32)
<<<<<<< HEAD
    logger.add_hook(lambda logger, data: graph.plot_metrics(logger, log_transfers=False), feature="epoch", freq=32)
    # logger.add_hook(lambda logger, data: graph.update_paths(logger), feature="epoch", freq=32)

    graph.plot_estimates(logger)
    graph.plot_metrics(logger, log_transfers=False)
    # graph.plot_paths(logger, 
    #     dest_tasks=[tasks.normal, tasks.depth_zbuffer, tasks.principal_curvature], 
    #     show_images=True
    # )

    for epochs in range(0, 256):
        logger.update("epoch", epochs)

        free_energy = graph.free_energy(sample=8)
        logger.update("energy", free_energy)

        graph.estimates.step(free_energy) # if you uncomment this it eventually runs out of mem at epoch 15
        logger.step()
=======
    logger.add_hook(lambda logger, data: graph.update_paths(logger), feature="epoch", freq=32)


    graph.plot_estimates(logger)
    graph.plot_paths(logger, 
        dest_tasks=[tasks.normal,], 
        show_images=True,
        max_len=2,
    )

    for epochs in range(0, 8000):
        logger.update("epoch", epochs)

        free_energy = graph.free_energy(sample=8)
        # print (epochs, free_energy)
        graph.step(free_energy)
        logger.update("energy", free_energy)

        # if epochs % 32 == 0:
        #     for task in graph.tasks:
        #         print (f"{task}, {graph.estimates[task.name].mean()}, {graph.stds[task.name].mean()}")
>>>>>>> robust statistics + mle meanvariance estimator

if __name__ == "__main__":
    Fire(main)


