
import os, sys, math, random, itertools
import numpy as np
from time import sleep
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import TaskDataset
from datasets import load_sintel_train_val_test, load_video_games, load_ood

from fire import Fire
from modules.unet import UNetOld

from transfers import finetuned_transfers
from task_configs import get_task

import IPython

def main():

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)

    resize = 256
    ood_images = load_ood()[0]
    tasks = [get_task(name) for name in ['rgb', 'normal', 'principal_curvature', 'depth_zbuffer', 'sobel_edges', 'reshading', 'keypoints3d', 'keypoints2d']]

    test_loader = torch.utils.data.DataLoader(
        TaskDataset(['almena'], tasks),
        batch_size=64,
        num_workers=12,
        shuffle=False,
        pin_memory=True
    )
    imgs = list(itertools.islice(test_loader, 1))[0]
    gt = {tasks[i].name:batch.cuda() for i, batch in enumerate(imgs)}
    num_plot = 4

    logger.images(ood_images, f"x", nrow=2, resize=resize)
    edges = finetuned_transfers

    def get_nbrs(task, edges):
        res = []
        for e in edges:
            if task == e.src_task: 
                res.append(e)
        return res

    max_depth = 10
    mse_dict = defaultdict(list)

    def search_small(x, task, prefix, visited, depth, endpoint):

        if task.name == 'normal':
            interleave = torch.stack([val for pair in zip(x[:num_plot], gt[task.name][:num_plot]) for val in pair])
            logger.images(interleave.clamp(max=1, min=0), prefix, nrow=2, resize=resize)
            mse, _ = task.loss_func(x, gt[task.name])
            mse_dict[task.name].append((mse.detach().data.cpu().numpy(), prefix))

        for transfer in get_nbrs(task, edges):
            preds = transfer(x)
            next_prefix = f'{transfer.name}({prefix})'
            print(f"{transfer.src_task.name}2{transfer.dest_task.name}", next_prefix)
            
            if transfer.dest_task.name not in visited:
                visited.add(transfer.dest_task.name)
                res = search_small(preds, transfer.dest_task, next_prefix, visited, depth+1, endpoint)
                visited.remove(transfer.dest_task.name)

        return endpoint == task

    def search_full(x, task, prefix, visited, depth, endpoint):
        for transfer in get_nbrs(task, edges):
            preds = transfer(x)
            next_prefix = f'{transfer.name}({prefix})'
            print(f"{transfer.src_task.name}2{transfer.dest_task.name}", next_prefix)
            if transfer.dest_task.name == 'normal':
                interleave = torch.stack([val for pair in zip(preds[:num_plot], gt[transfer.dest_task.name][:num_plot]) for val in pair])
                logger.images(interleave.clamp(max=1, min=0), next_prefix, nrow=2, resize=resize)
                mse, _ = task.loss_func(preds, gt[transfer.dest_task.name])
                mse_dict[transfer.dest_task.name].append((mse.detach().data.cpu().numpy(), next_prefix))
            if transfer.dest_task.name not in visited:
                visited.add(transfer.dest_task.name)
                res = search_full(preds, transfer.dest_task, next_prefix, visited, depth+1, endpoint)
                visited.remove(transfer.dest_task.name)

        return endpoint == task

    def search(x, task, prefix, visited, depth):
        for transfer in get_nbrs(task, edges):
            preds = transfer(x)
            next_prefix = f'{transfer.name}({prefix})'
            print(f"{transfer.src_task.name}2{transfer.dest_task.name}", next_prefix)
            if transfer.dest_task.name == 'normal':
                logger.images(preds.clamp(max=1, min=0), next_prefix, nrow=2, resize=resize)
            if transfer.dest_task.name not in visited:
                visited.add(transfer.dest_task.name)
                res = search(preds, transfer.dest_task, next_prefix, visited, depth+1)
                visited.remove(transfer.dest_task.name)

    with torch.no_grad():
        # search_full(gt['rgb'], TASK_MAP['rgb'], 'x', set('rgb'), 1, TASK_MAP['normal'])
        search(ood_images, get_task('rgb'), 'x', set('rgb'), 1)
    
    for name, mse_list in mse_dict.items():
        mse_list.sort()
        print(name)
        print(mse_list)
        if len(mse_list) == 1: mse_list.append((0, '-'))
        rownames = [pair[1] for pair in mse_list]
        data = [pair[0] for pair in mse_list]
        print(data, rownames)
        logger.bar(data, f'{name}_path_mse', opts={'rownames':rownames})
    

if __name__ == "__main__":
    Fire(main)
