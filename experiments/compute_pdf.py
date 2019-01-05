
import os, sys, math, random, itertools, pickle, yaml
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
from datasets import TaskDataset, load_all
from graph import TaskGraph

from fire import Fire
import IPython


def main():

    task_list = [
        tasks.rgb, 
        tasks.normal, 
        tasks.principal_curvature, 
        tasks.reshading, 
        tasks.sobel_edges,
        tasks.keypoints3d,
        tasks.keypoints2d,
        tasks.depth_zbuffer,
        tasks.edge_occlusion,
    ]

    train_loader = load_all(task_list, buildings=["almena"], batch_size=128)
    divergences = defaultdict(list)
    variance_data = {}

    for task_data in itertools.islice(train_loader, 10):
        for task, data in zip(task_list, task_data):
            divergence = torch.mean((data - data.mean(dim=0, keepdim=True))**2)*128.0/127.0
            divergences[task.name] += [divergence.data.cpu().numpy()]
            print (task.name, divergence)

    for task in task_list:
        variance_data[task.name] = float(np.mean(divergences[task.name]))
    
    with open(f"{MODELS_DIR}/variances.txt", 'w') as outfile:
        yaml.dump(variance_data, outfile)
        

if __name__ == "__main__":
    Fire(main)
