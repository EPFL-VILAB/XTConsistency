
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
        tasks.keypoints2d,
    ]

    reality = RealityTask('almena', 
        dataset=TaskDataset(buildings=['almena'], 
            tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature, tasks.depth_zbuffer]
        ),
        tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature, tasks.depth_zbuffer],
        batch_size=8
    )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        anchored_tasks=[reality, tasks.rgb],
        reality=reality,
        batch_size=8,
        edges_exclude=[
            ('almena', 'normal'),
            ('almena', 'principal_curvature'),
            ('almena', 'depth_zbuffer'),
            # ('rgb', 'keypoints3d'),
            # ('rgb', 'edge_occlusion'),
        ],
        initialize_first_order=False,
    )

    graph.p.compile(torch.optim.Adam, lr=4e-2)
    graph.estimates.compile(torch.optim.Adam, lr=1e-2)

    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="energy", freq=16)
    logger.add_hook(lambda logger, data: logger.plot(data["energy"], "free_energy"), feature="energy", freq=100)
    logger.add_hook(lambda logger, data: graph.plot_estimates(logger), feature="epoch", freq=32)
    # logger.add_hook(lambda logger, data: graph.update_paths(logger), feature="epoch", freq=32)

    # graph.plot_estimates(logger)
    # graph.plot_paths(logger, 
    #     dest_tasks=[tasks.normal, tasks.depth_zbuffer, tasks.principal_curvature], 
    #     show_images=False
    # )

    logger.add_hook(lambda logger, data: logger.plot(data["loss"], "loss"), feature="loss", freq=100)
    logger.add_hook(lambda logger, data: logger.plot(data["f(y), f(y_hat)"], "f(y), f(y_hat)"), feature="f(y), f(y_hat)", freq=100)
    logger.add_hook(lambda logger, data: logger.plot(data["F(f(y)), F(f(y_hat))"], "F(f(y)), F(f(y_hat))"), feature="F(f(y)), F(f(y_hat))", freq=100)

    for epochs in range(0, 750):
        logger.update("epoch", epochs)

        loss, (curv_loss, cycle_loss) = graph.cycle_loss_test()
        graph.estimates.step(loss)
        logger.update("loss", loss)
        logger.update("f(y), f(y_hat)", curv_loss)
        logger.update("F(f(y)), F(f(y_hat))", cycle_loss)
        # free_energy = graph.free_energy(sample=4)
        # print (epochs, free_energy)
        # graph.estimates.step(free_energy) # if you uncomment this it eventually runs out of mem at epoch 15
        # logger.update("energy", free_energy)
        logger.step()

if __name__ == "__main__":
    Fire(main)


