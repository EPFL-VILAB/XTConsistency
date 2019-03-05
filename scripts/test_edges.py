
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
        tasks.sobel_edges,
    ]

    reality = RealityTask('almena', 
        dataset=TaskDataset(
            buildings=['almena'],
            tasks=task_list,
        ),
        tasks=task_list,
        batch_size=8
    )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        batch_size=28
    )

    x = graph.edge_map[('rgb', 'sobel_edges')] (reality.task_data[tasks.rgb])
    y = (reality.task_data[tasks.sobel_edges])
    logger.images_grouped([x, y], "rgb, sobel_edges", resize=256)


if __name__ == "__main__":
    Fire(main)
