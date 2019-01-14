
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
from transfers import functional_transfers, TRANSFER_MAP
from datasets import TaskDataset
from graph import TaskGraph
from fire import Fire
import IPython


def main():

    task_list = [
        tasks.rgb,
        tasks.normal,
        tasks.principal_curvature,
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
        ],
        initialize_first_order=False,
    )

    graph.p.compile(torch.optim.Adam, lr=4e-2)
    graph.estimates.compile(torch.optim.Adam, lr=1e-2)

    graph.estimates['principal_curvature'].data = functional_transfers.f(reality.task_data[tasks.normal].to(DEVICE)).data

    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=16)
    logger.add_hook(lambda logger, data: graph.plot_estimates(logger), feature="epoch", freq=32)

    logger.add_hook(lambda logger, data: logger.plot(data["loss"], "loss"), feature="loss", freq=100)
    logger.add_hook(lambda logger, data: logger.plot(data["consistency"], "f(y), f(y_hat)"), feature="consistency", freq=100)
    logger.add_hook(lambda logger, data: logger.plot(data["cycle_step1"], "f(y), z"), feature="cycle_step1", freq=100)
    logger.add_hook(lambda logger, data: logger.plot(data["cycle_step2"], "f(z), y"), feature="cycle_step2", freq=100)
    logger.add_hook(lambda logger, data: logger.plot(data["normal_error"], "y, y_hat"), feature="normal_error", freq=100)

    logger.add_hook(lambda logger, data: graph.update_paths(logger), feature="epoch", freq=32)

    graph.plot_estimates(logger)
    graph.plot_paths(logger,
        dest_tasks=[tasks.normal, tasks.depth_zbuffer, tasks.principal_curvature], 
        show_images=False, max_len=2,
    )

    for epochs in range(0, 750):
        logger.update("epoch", epochs)
        loss, (consistency, cycle_step1, cycle_step2, normal_error) = graph.cycle_synthesis_test()
        graph.estimates.step(loss)

        logger.update("loss", loss)
        logger.update("consistency", consistency)
        logger.update("cycle_step1", cycle_step1)
        logger.update("cycle_step2", cycle_step2)
        logger.update("normal_error", normal_error)


if __name__ == "__main__":
    Fire(main)
