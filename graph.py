
import os, sys, math, random, itertools
from collections import namedtuple, defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, task_map, tasks, get_model
from transfers import Transfer, pretrained_transfers



class TaskGraph(object):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(self, 
            models_dir=MODELS_DIR, 
            tasks=tasks, task_filter=[tasks.segment_semantic, tasks.class_scene]
        ):
        super().__init__()
        self.models_dir = models_dir
        self.estimates = {}
        self.tasks = list(set(tasks) - set(task_filter))
        self.load()
    
    def load(self, path=None, 
            task_filter=[tasks.segment_semantic, tasks.class_scene], 
            transfer_set=None,
        ):

        self.tasks = list(set(tasks) - set(task_filter))
        self.edges, self.adj = {}, defaultdict(list)

        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task.name, dest_task.name)
            if transfer_set is not None and key not in transfer_set: continue
            

            model_type, model_path = pretrained_transfers.get(key, (None, None))
            if model_path is not None: model_path = model_path.split('/')[-1]
            model_type = model_type or partial(get_model, src_task.name, dest_task.name)
            model_path = (path or self.models_dir) + "/" + (model_path or f"{src_task}2{dest_task}.pth")
            print (model_path)
            if not os.path.isfile(model_path):
                print (f"No transfer available for {src_task}, {dest_task}")
                continue
 
            self.edges[key] = Transfer(src_task, dest_task, model_type=model_type, path=model_path)
            self.adj[src_task.name] += [self.edges[key]]

    def save(self, directory):
        raise NotImplementedError()








if __name__ == "__main__":
    graph = TaskGraph()
    distribution = UniformPathDistribution(graph, max_path_length=3)
    path = distribution.sample()
    print ("Path: ", path)
    graph.eval_path(torch.randn(1, 3, 256, 256), path=path)
    
