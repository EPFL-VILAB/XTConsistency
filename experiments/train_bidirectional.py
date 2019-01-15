
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from plotting import *
from functional import get_functional_loss
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImagePairDataset, load_train_val, load_test, load_ood
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

from transfers import functional_transfers
from task_configs import tasks
from graph import TaskGraph
import IPython


def main(pretrained=False, batch_size=12, fast=False, **kwargs):
    # if fast: batch_size = 8
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

    # MODEL
    model = functional_transfers.n.load_model() if pretrained else DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=(3e-5 if pretrained else 3e-4), weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)
    
    # TRANSFER LOADING
    graph = TaskGraph(tasks=task_list, batch_size=1)
    in_transfers = [t for t in graph.in_adj[tasks.normal] if t.name != 'n']
    out_transfers = [t for t in graph.edges if t.src_task == tasks.normal]
    
    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
    logger.add_hook(lambda logger, data: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)
    logger.add_hook(lambda logger, data: scheduler.step(), feature="epoch", freq=1)
    logger.add_hook(partial(jointplot, loss_type="mse_loss"), feature="val_mse_loss", freq=1)
    losses = defaultdict(list)
    for t in in_transfers + out_transfers:
        losses[t.name].append(torch.tensor(0).to(DEVICE))
        logger.update(t.name, losses[t.name][-1])
        logger.add_hook(lambda logger, data, name=t.name: logger.plot(data[name], name), feature=t.name, freq=1)

    # DATA LOADING
    
    ood_images = load_ood()
    data_loaders = load_train_val(task_list, batch_size=batch_size, val_tasks=['rgb', 'normal'],
        train_buildings=(["almena"] if fast else None), val_buildings=(["almena"] if fast else None))
    train_loader, val_loader, train_step, val_step = data_loaders
    test_set, test_images = load_test("rgb", "normal")
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    for t in in_transfers:
        _, gt_images = load_test(t.src_task.name, t.dest_task.name)
        logger.images(t(gt_images).clamp(min=0, max=1), f"{t.name}_incoming", resize=256)

    # LOSS FUNCTION
    def loss_func(pred, target):
        target = {task_list[i+1].name : target[i] for i in range(len(target))}
        mse, _ = tasks.normal.norm(pred, target['normal'])
        total = 0
        for transfer in in_transfers:
            with torch.no_grad():
                gt = transfer(target[transfer.src_task.name])
            loss, val = transfer.dest_task.norm(pred, gt)
            losses[transfer.name].append(val[0].data.cpu())
            total += loss
        for transfer in out_transfers:
            loss, val = transfer.dest_task.norm(transfer(pred), target[transfer.src_task.name])
            losses[transfer.name].append(val[0].data.cpu())
            total += loss
        return mse + total, (mse.detach(),)
    
    # TRAINING
    for epochs in range(0, 800):
        preds_name = "start_preds" if epochs == 0 and pretrained else "preds"
        ood_name = "start_ood" if epochs == 0 and pretrained else "ood"
        plot_images(model, logger, test_set, dest_task="normal", ood_images=ood_images, 
            preds_name=preds_name, ood_name=ood_name
        )
        logger.update("epoch", epochs)
        logger.step()

        train_set = itertools.islice(train_loader, 5 if fast else train_step)
        val_set = itertools.islice(val_loader, 5 if fast else val_step)

        (val_mse_data,) = model.predict_with_metrics(val_set, loss_fn=tasks.normal.norm, logger=logger)
        (train_mse_data,) = model.fit_with_metrics(train_set, loss_fn=loss_func, logger=logger)
        logger.update("train_mse_loss", torch.mean(torch.tensor(train_mse_data)))
        logger.update("val_mse_loss", torch.mean(torch.tensor(val_mse_data)))
        for name, data in losses.items():
            logger.update(name, torch.mean(torch.tensor(data[1:][-train_step:])))
        # run_eval_suite(model, logger, sample=160)


if __name__ == "__main__":
    Fire(main)
