
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
from datasets import TaskDataset, ImagePairDataset
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
        tasks.reshading,
        tasks.edge_occlusion,
        tasks.keypoints3d,
        tasks.keypoints2d,
    ]

    reality = RealityTask('ood', 
        dataset=ImagePairDataset(data_dir=OOD_DIR, resize=(256, 256)),
        tasks=[tasks.rgb, tasks.rgb],
        batch_size=28
    )

    # reality = RealityTask('almena', 
    #     dataset=TaskDataset(
    #         buildings=['almena'],
    #         tasks=task_list,
    #     ),
    #     tasks=task_list,
    #     batch_size=8
    # )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        batch_size=28
    )

    task = tasks.rgb
    images = [reality.task_data[task]]
    sources = [task.name]
        
    for _, edge in sorted(((edge.dest_task.name, edge) for edge in graph.adj[task])):
        if isinstance(edge.src_task, RealityTask): continue

        reality.task_data[edge.src_task]
        x = edge(reality.task_data[edge.src_task])

        if edge.dest_task != tasks.normal:
            edge2 = graph.edge_map[(edge.dest_task.name, tasks.normal.name)]
            x = edge2(x)

        images.append(x.clamp(min=0, max=1))
        sources.append(edge.dest_task.name)

    logger.images_grouped(images, ", ".join(sources), resize=256)

if __name__ == "__main__":
    Fire(main)
