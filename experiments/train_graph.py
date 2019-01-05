
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

from fire import Fire
import IPython


def main():

    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="energy", freq=16)
    logger.add_hook(lambda logger, data: logger.plot(data["energy"], "free_energy"), feature="energy", freq=100)

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
        # edges=[
        #     ('normal', 'rgb'),
        #     ('normal', 'principal_curvature'),
        #     ('normal', 'sobel_edges'),
        #     ('normal', 'depth_zbuffer'),
        #     ('depth_zbuffer', 'normal'),
        #     ('sobel_edges', 'normal'),
        #     ('principal_curvature', 'normal'),
        #     ('rgb', 'normal'),
        # ],
        edges_exclude=[
            ('almena', 'normal'),  #remove all GT for normals
            # ('almena', 'principal_curvature'),  #remove all GT for normals
            ('almena', 'depth_zbuffer'),  #remove all GT for normals
            ('almena', 'sobel_edges'),  #remove all GT for normals
            ('rgb', 'normal'), #remove all GT substitutes (pretraining?)
            ('rgb', 'principal_curvature'), #remove all GT substitutes (pretraining?)
            ('rgb', 'depth_zbuffer'), #remove all GT substitutes (pretraining?)
            ('rgb', 'sobel_edges'), #remove all GT substitutes (pretraining?)
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

    # graph.estimates['normal'].data = reality.task_data[tasks.normal].to(DEVICE)
    # reality.step()

    # graph.estimates['depth_zbuffer'].data = reality.task_data[tasks.depth_zbuffer].to(DEVICE)
    # reality.step()

    # graph.estimates['sobel_edges'].data = reality.task_data[tasks.sobel_edges].to(DEVICE)
    # reality.step()

    # graph.estimates['normal'].data = reality.task_data[tasks.normal].to(DEVICE)
    # graph.estimates['depth_zbuffer'].data = reality.task_data[tasks.depth_zbuffer].to(DEVICE)
    graph.estimates['rgb'].data = reality.task_data[tasks.rgb].to(DEVICE)
    graph.estimates['principal_curvature'].data = reality.task_data[tasks.principal_curvature].to(DEVICE)
    # graph.estimates['sobel_edges'].data = reality.task_data[tasks.sobel_edges].to(DEVICE)

    for task in graph.tasks:
        task.plot_func(graph.estimates[task.name], task.name, logger)

    for epochs in range(0, 4000):
        free_energy = graph.free_energy(sample=12)
        # graph.p.step(free_energy)
        graph.estimates.step(free_energy)
        logger.update("energy", free_energy)

        if epochs % 32 == 0:
            # for task in graph.tasks:
            #     print (f"{task.name}: {graph.prob(task)}")

            for task in graph.tasks:
                task.plot_func(graph.estimates[task.name], task.name, logger)
    
if __name__ == "__main__":
    Fire(main)

