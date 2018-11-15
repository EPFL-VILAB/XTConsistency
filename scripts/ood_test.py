
import os, sys, math, random, itertools
import numpy as np
import pickle
from time import sleep
from collections import defaultdict


import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset, ImageDataset, GeneralTaskLoader

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from modules.percep_nets import DenseNet, DeepNet, BaseNet
from modules.depth_nets import UNetDepth
from sklearn.model_selection import train_test_split
from fire import Fire

from transfers import functional_transfers
from task_configs import TASK_MAP
import IPython

def main():

    model = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/mixing_percepcurv_norm.pth")
    model.compile(torch.optim.Adam, lr=5e-4, weight_decay=2e-6, amsgrad=True)

    print (model.forward(torch.randn(8, 3, 256, 256)).shape)
    print (model.forward(torch.randn(16, 3, 256, 256)).shape)
    print (model.forward(torch.randn(32, 3, 256, 256)).shape)
    
    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        mse = F.mse_loss(pred*mask.float(), target*mask.float())
        return mse, (mse.detach(),)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=25)
    resize = 256
    data_dir = f'{SHARED_DIR}/ood_images'
    # data_dir = f'{BASE_DIR}/data/taskonomy3/almena_rgb/rgb/'
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir=data_dir, resize=(resize, resize)),
        batch_size=10,
        num_workers=10,
        shuffle=False,
        pin_memory=True
    )
    ood_images = list(itertools.islice(ood_loader, 1))[0]
    tasks = [TASK_MAP[name] for name in ['rgb', 'normal', 'principal_curvature', 'depth_zbuffer', 'sobel_edges', 'reshading', 'keypoints3d', 'keypoints2d']]
    # tasks = [task for name, task in TASK_MAP.items()]

    test_loader = torch.utils.data.DataLoader(
        GeneralTaskLoader(['almena'], tasks),
        batch_size=64,
        num_workers=12,
        shuffle=False,
        pin_memory=True
    )
    imgs = list(itertools.islice(test_loader, 1))[0]
    gt = {tasks[i].name:batch.cuda() for i, batch in enumerate(imgs)}
    num_plot = 4

    logger.images(gt['rgb'][:4], f"x", nrow=1, resize=resize)
    # logger.images(ood_images, f"x", nrow=1, resize=resize)
    edges = functional_transfers

    def get_nbrs(task, edges):
        res = []
        for e in edges:
            if task == e.src_task: 
                res.append(e)
        return res
    IPython.embed()
    max_depth = 10
    mse_dict = defaultdict(list)
    for t in functional_transfers:
        print(t.name)
        sleep(1)
        t.load_model()

    def search_small(x, task, prefix, visited, depth, endpoint):

        if task.name == 'normal':
            interleave = torch.stack([val for pair in zip(x[:num_plot], gt[task.name][:num_plot]) for val in pair])
            logger.images(interleave.clamp(max=1, min=0), prefix, nrow=2, resize=resize)
            mse, _ = task.loss_func(preds, gt[task.name])
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
        search2(gt['rgb'], TASK_MAP['rgb'], 'x', set('rgb'), 1, TASK_MAP['normal'])
        # search(ood_images, TASK_MAP['rgb'], 'x', set('rgb'), 1)
    
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
