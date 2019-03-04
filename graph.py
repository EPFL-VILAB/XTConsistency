import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from transfers import Transfer, RealityTransfer, get_named_transfer
import transforms


class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(
        self, tasks=tasks, edges=None, edges_exclude=None, 
        pretrained=True, finetuned=False,
        reality=[], task_filter=[tasks.segment_semantic, tasks.class_scene],
    ):

        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map, self.reality = {}, reality

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            if edges is not None and key not in edges: continue
            if edges_exclude is not None and key in edges_exclude: continue
            if src_task == dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            if src_task.name != src_task.kind: continue

            transfer = get_named_transfer(Transfer(src_task, dest_task, 
                pretrained=pretrained, finetuned=finetuned
            ))
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                transfer = RealityTransfer(src_task, dest_task)

            if transfer.model_type is None: continue

            self.edges += [transfer]
            self.adj[src_task.kind] += [transfer]
            self.in_adj[dest_task.name] += [transfer]
            self.edge_map[str((src_task.kind, dest_task.name))] = transfer
            transfer.load_model()

        self.edge_map = nn.ModuleDict(self.edge_map)

    def edge(self, src_task, dest_task):
        return self.edge_map[str((src_task.kind, dest_task.name))]

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
                x = cache.get(tuple(path[0:(i+1)]), 
                    self.edge(path[i-1], path[i])(x)
                )
            except KeyError:
                return None
            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x



