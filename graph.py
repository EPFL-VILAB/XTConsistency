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
from transfers import Transfer, RealityTransfer, get_transfer_name
import transforms

from modules.gan_dis import GanDisNet

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
        self.tasks += [task.base for task in self.tasks if hasattr(task, "base")]
        self.edge_list, self.edge_list_exclude = edges, edges_exclude
        self.pretrained, self.finetuned = pretrained, finetuned
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map, self.reality = {}, reality
        print('graph tasks', self.tasks)
        self.params = {}

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task, dest_task)
            if edges is not None and key not in edges: continue
            if edges_exclude is not None and key in edges_exclude: continue
            if src_task == dest_task: continue
            if isinstance(dest_task, RealityTask): continue
            print (src_task, dest_task)
            transfer = None
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                transfer = RealityTransfer(src_task, dest_task)
            else:
                transfer = Transfer(src_task, dest_task, 
                    pretrained=pretrained, finetuned=finetuned
                )
                transfer.name = get_transfer_name(transfer)
            if transfer.model_type is None: 
                continue
            print ("Added transfer", transfer)
            self.edges += [transfer]
            self.adj[src_task.name] += [transfer]
            self.in_adj[dest_task.name] += [transfer]
            self.edge_map[str((src_task.name, dest_task.name))] = transfer
            if isinstance(transfer, nn.Module):
                self.params[str((src_task.name, dest_task.name))] = transfer
                try:
                    transfer.load_model()
                except:
                    IPython.embed()

        self.params = nn.ModuleDict(self.params)

    def edge(self, src_task, dest_task):
        key1 = str((src_task.name, dest_task.name))
        key2 = str((src_task.kind, dest_task.kind))
        if key1 in self.edge_map: return self.edge_map[key1]
        return self.edge_map[key2]

    def sample_path(self, path, reality=None, use_cache=False, cache={}):
        path = [reality or self.reality[0]] + path
        x = None
        for i in range(1, len(path)):
            try:
                x = cache.get(tuple(path[0:(i+1)]), 
                    self.edge(path[i-1], path[i])(x)
                )
            except KeyError:
                print ("Failed")
                pdb.set_trace()
                IPython.embed()
                return None
            if use_cache: cache[tuple(path[0:(i+1)])] = x
        return x

    def save(self, weights_file=None, weights_dir=None):

        ### TODO: save optimizers here too
        if weights_file:
            torch.save({
                key: model.state_dict() for key, model in self.edge_map.items() \
                if not isinstance(model, RealityTransfer)
            }, weights_file)

        if weights_dir:
            os.makedirs(weights_dir, exist_ok=True)
            for key, model in self.edge_map.items():
                if isinstance(model, RealityTransfer): continue
                model.model.save(f"{weights_dir}/{model.name}.pth")


    def load_weights(self, weights_file=None):
        for key, state_dict in torch.load(weights_file).items():
            if key in self.edge_map:
                self.edge_map[key].load_state_dict(state_dict)


class Discriminator(object):
    def __init__(self, loss_config):
        super(Discriminator, self).__init__()
        self.discriminator_dict = {}

        dis_net = GanDisNet()
        dis_net.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
        for reality_gan in loss_config:
            for gan_term in loss_config[reality_gan]:
                #self.discriminator_dict[gan_term[0]+gan_term[1]] = GanDisNet()
                #self.discriminator_dict[gan_term[0]+gan_term[1]].compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
                self.discriminator_dict[gan_term[0]+gan_term[1]] = dis_net

    def train(self):
        for _, discriminator in self.discriminator_dict.items():
            discriminator.train()

    def eval(self):
        for _, discriminator in self.discriminator_dict.items():
            discriminator.eval()

    def step(self, loss):
        len1 = len(self.discriminator_dict)
        k = 0
        for dis_key, discriminator in self.discriminator_dict.items():
            if k == len1-1:
                discriminator.step(-loss['disgan'+dis_key])
            else:
                discriminator.step(-loss['disgan'+dis_key], retain_graph=True)
            k+=1

    def __getitem__(self, idx):
        return self.discriminator_dict[idx]

    def save(self, weights_file=None):
        weights = {
            key: model.state_dict() for key, model in self.discriminator_dict.items()
        }
        optimizer = {
            key: model.optimizer.state_dict() for key, model in self.discriminator_dict.items()
        }
        torch.save({'weights':weights, 'optimizer':optimizer}, weights_file)


    def load_weights(self, weights_file=None):
        file_ = torch.load(weights_file)
        for key, state_dict in flie_['weights'].items():
            if key in self.discriminator_dict:
                self.discriminator_dict[key].load_state_dict(state_dict)
        for key, state_dict in flie_['optimizer'].items():
            if key in self.discriminator_dict:
                self.discriminator_dict[key].optimizer.load_state_dict(state_dict)

