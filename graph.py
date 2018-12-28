
import os, sys, math, random, itertools
from collections import namedtuple, defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from transfers import Transfer, pretrained_transfers



class TaskGraph(nn.Module):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(self, 
            tasks=tasks, task_filter=[tasks.segment_semantic, tasks.class_scene]
        ):
        super().__init__()
        self.models_dir = models_dir
        self.tasks = list(set(tasks) - set(task_filter))
        self.edges, self.adj = [], defaultdict(list)

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task.name, dest_task.name)
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                self.edges[key] = RealityTransfer(src_task, dest_task)
                self.adj[src_task.name] += [self.edges[key]]

            transfer = Transfer(src_task, dest_task, model_type=model_type, path=model_path)
            if transfer.model_type is None: continue
            self.edges += [transfer]
            self.adj[src_task] += [self.edges[key]]

        self.estimates = nn.ParameterDict({
            task.name: nn.Parameter(torch.randn(*task.shape).requires_grad_(True)) for task in tasks
        })
        self.parameters = {} #differentiable weightings in free energy formula, not currently set right now

    def free_energy(self):

        return sum(
            (
                transfer.dest_task.norm(
                    transfer(self.estimates[transfer.src_task]), 
                    self.estimates[transfer.dest_task]
                ) for transfer in self.edges
            )
        )




if __name__ == "__main__":
    graph = TaskGraph()
    print (graph.edges)
    print (graph.free_energy())
    
