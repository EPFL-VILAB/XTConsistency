
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
from datasets import TaskDataset, SintelDataset
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
        # tasks.edge_occlusion,
        # tasks.keypoints3d,
        tasks.keypoints2d,
    ]
    # tasks.depth_zbuffer.image_transform = tasks.depth_zbuffer.sintel_depth.image_transform

    reality = RealityTask('sintel', 
        dataset=SintelDataset(buildings=None, tasks=[tasks.rgb, tasks.normal]),
        tasks=[tasks.rgb, tasks.normal],
        batch_size=48
    )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        anchored_tasks=[reality, tasks.rgb],
        reality=reality,
        batch_size=48,
        edges_exclude=[
            ('sintel', 'normal'),
            ('sintel', 'depth_zbuffer'),
        ],
        initialize_first_order=True,
    )
    print (graph.edges)
    # assume all estimates are rgb2x
    # Show that you can get a gain with the appropriate neigbor weighting (either GT or rgb2y)
    # quantify this
    
    for task in [tasks.normal,]:
        Y = graph.estimate(task)
        Y_hat = reality.task_data[task].to(DEVICE).detach()
        mse_orig = task.norm(Y, Y_hat)[0].data.cpu().numpy().mean()

        estimates = [
            transfer(graph.estimate(transfer.src_task)).detach() \
            for transfer in graph.in_adj[task] if not isinstance(transfer, RealityTask) and transfer.src_task is not tasks.rgb
        ]

        weights = WrapperModel(nn.ParameterList([
            nn.Parameter(torch.tensor(1.0).requires_grad_(True).to(DEVICE)) \
            for transfer in graph.in_adj[task] if not isinstance(transfer, RealityTask) and transfer.src_task is not tasks.rgb
        ]))
        weights.compile(torch.optim.Adam, lr=1e-2)

        mse = None
        for i in range(0, 1000):
            average = sum((weight*estimate for weight, estimate in zip(weights, estimates)))/sum(weights)
            mse = task.norm(average, reality.task_data[task].to(DEVICE).detach())[0]
            weights.step(mse)

        mse = mse.data.cpu().numpy().mean()
        print (f"Train set {task}: rgb2x={mse_orig}, best case={mse}")

        for i in range(0, len(estimates)):
            print (f"{graph.in_adj[task][i]}: {weights[i]}")
        
        reality.step()
        graph.init_params()

        Y = graph.estimate(task)
        Y_hat = reality.task_data[task].to(DEVICE).detach()
        mse_orig = task.norm(Y, Y_hat)[0].data.cpu().numpy().mean()

        estimates = [
            transfer(graph.estimate(transfer.src_task)).detach() \
            for transfer in graph.in_adj[task] if not isinstance(transfer, RealityTask)
        ]

        average = sum((weight*estimate for weight, estimate in zip(weights, estimates)))/sum(weights)
        mse = task.norm(average, reality.task_data[task].to(DEVICE).detach())[0]

        print (f"Test set {task}: rgb2x={mse_orig}, best case={mse}")

        logger.images()

if __name__ == "__main__":
    Fire(main)


