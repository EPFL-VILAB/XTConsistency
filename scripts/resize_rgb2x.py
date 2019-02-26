
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
from torchvision import transforms

from fire import Fire
import IPython


def main():

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

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

    for task in task_list: 
        task.resize = 512
        task.load_image_transform()

    reality = RealityTask('almena', 
        dataset=TaskDataset(
            buildings=['almena'],
            tasks=task_list,
        ),
        tasks=task_list,
        batch_size=4,
        shuffle=False,
    )

    graph = TaskGraph(
        tasks=[reality, *task_list],
        batch_size=4
    )

    for resize in range(128, 512+1, 64):

        print ("Resize: ", resize)
        
        task = tasks.rgb
        inp = reality.task_data[task]
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize),
            transforms.ToTensor()
        ])
        inp = torch.stack([transform(x.cpu()) for x in inp])
        print (inp.shape)
        images = [inp]
        sources = [task.name]
        
        with torch.no_grad():
            for _, edge in sorted(((edge.dest_task.name, edge) for edge in graph.adj[task])):
                print (edge)
                if isinstance(edge.src_task, RealityTask): continue
                print ("running")
                x = edge(inp)
                print ("run")
                if edge.dest_task != tasks.normal:

                    edge2 = graph.edge_map[(edge.dest_task.name, tasks.normal.name)]
                    print ("2", edge2)
                    print (x.shape)
                    print (edge2)
                    x = edge2(x)
                    print ("done")

                images.append(x.clamp(min=0, max=1).data.cpu())
                sources.append(edge.dest_task.name)
                print (sources)

        logger.images_grouped(images, f"Resized {resize}: " + ", ".join(sources), resize=resize)

if __name__ == "__main__":
    Fire(main)
