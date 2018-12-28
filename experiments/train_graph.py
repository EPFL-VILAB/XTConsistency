
import os, sys, math, random, itertools, pickle
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
from losses import calculate_weight

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from fire import Fire

from functools import partial

from transfers import functional_transfers, TaskGraph

import IPython


class PathDistribution(object):
    def __init__(self, task_graph, start=None, max_depth=1):
        self.g = task_graph
        if start is None:
            start = self.g.edges[('rgb', 'normal')]
        self.start = start
        self.max_depth = 1

    def sample(self):
        raise NotImplementedError()

class UniformSampler(PathDistribution):

    def __init__(self, task_graph, **kwargs):
        super().__init__(task_graph, **kwargs)

    def sample(self):
        path, curr = [self.start], self.start.dest_task
        for i in range(self.max_depth):
            path.append(random.choice(self.g.tasks[curr.name]))
            curr = path[-1].dest_task
        return path

class UniformSampler(PathDistribution):

    def __init__(self, task_graph, **kwargs):
        super().__init__(task_graph, **kwargs)

    def sample(self):
        path, curr = [self.start], self.start.dest_task
        for i in range(self.max_depth):
            path.append(random.choice(self.g.tasks[curr.name]))
            curr = path[-1].dest_task
        return path

class PathModel(TrainableModel):

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler
        transfer.load_model()
        self.model = transfer.model
    
    def forward(self, x):
        path = sampler.sample()
        with torch.no_grad():
            for f in path[:i]:
                x = f(x)
        return self.model(x)

    def loss(self, pred, target):
        loss1, _ = transfer.dest_task.loss_func(pred, target)
        loss2 = torch.tensor(0.0, device=pred.device)
        if i+1 < len(path):
            loss2, _ = path[i+1].dest_task.loss_func(pred, target)
        return loss1 + loss2, (loss1.detach(), loss2.detach()) 


def main(max_depth=1, mode="mixing", pretrained=False, **kwargs):

    # GET TASK GRAPH
    g = TaskGraph()

    # MODEL
    model = g.edges[('rgb', 'normal')].load_model() if pretrained else DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    # Path Sampler
    sampler = UniformSampler(g)

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
    logger.add_hook(lambda logger, data: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)
    logger.add_hook(lambda logger, data: scheduler.step(), feature="epoch", freq=1)
    logger_hooks = set()
    def save_models(logger, data):
        for edge in g.edges.values():
            if edge.model is not None:
                edge.load_model().save(f'{RESULTS_DIR}/{edge.src_task}2{edge.dest_task}.pth')
        with open(f"{RESULTS_DIR}/logger.p", "wb") as fp:
             pickle.dump(logger.data, fp)
    logger.add_hook(save_models, feature="epoch", freq=3)

    # DATA LOADING
    ood_images = load_ood()
    train_loader, val_loader, train_step, val_step = load_train_val("rgb", "normal", batch_size=48)
        # train_buildings=['almena'], val_buildings=['almena'])
    test_set, test_images = load_test("rgb", "normal")
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)

    for edge in g.edges.values():
        edge.checkpoint = False

    # TRAINING
    for epochs in range(0, 800):
        preds_name = "start_preds" if epochs == 0 and pretrained else "preds"
        ood_name = "start_ood" if epochs == 0 and pretrained else "ood"
        plot_images(model, logger, test_set, dest_task="normal", ood_images=ood_images, 
            preds_name=preds_name, ood_name=ood_name, show_masks=True
        )
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        val_set = itertools.islice(val_loader, val_step)
        train_metrics, val_metrics = defaultdict(list), defaultdict(list)

        for x, y in val_set:
            with torch.no_grad():
                path = sampler.sample()
                preds = path[0](x)
                gt = y.detach().to(preds.device) 
                for i, transfer in enumerate(path):
                    if i == 0: continue
                    gt_mse, _ = transfer.src_task.norm(preds, gt)
                    next_gt, next_preds = path[i](gt), path[i](preds)
                    percep_mse, _ = transfer.dest_task.norm(next_preds, next_gt)
                    loss = 0.5 * gt_mse + 0.5 * percep_mse
                    gt, preds = next_gt, next_preds
                    val_metrics[f'{path[i-1].name}_mse'].append(gt_mse.detach())
                    val_metrics[f'{transfer.name}_percep'].append(percep_mse.detach())
                    val_metrics['mse_weight'].append(0.5)
                logger.update("loss", float(loss))

        for x, y in train_set:
            path = sampler.sample()
            x.requires_grad = True
            preds = path[0](x)
            gt = y.detach().to(preds.device)
            for i, transfer in enumerate(path):
                if i == 0: continue
                
                gt_mse, _ = transfer.src_task.norm(preds, gt)
                with torch.no_grad(): next_gt = path[i](gt).detach()
                next_preds = path[i](preds)
                
                percep_mse, _ = transfer.dest_task.norm(next_preds, next_gt)
                if i < len(path)-1:
                    gt, preds = next_gt, path[i](preds.detach())
                
                curr_model = path[i-1].load_model()
                c1, c2 = calculate_weight(curr_model, gt_mse, percep_mse)
                
                loss = c1 * gt_mse + c2 * percep_mse
                curr_model.train(True)
                loss.backward()
                curr_model.optimizer.step()
                curr_model.zero_grad()
                curr_model.optimizer.zero_grad()
                train_metrics[f'{path[i-1].name}_mse'].append(gt_mse.detach())
                train_metrics[f'{transfer.name}_percep'].append(percep_mse.detach())
                train_metrics['mse_weight'].append(c1)
            logger.update("loss", float(loss))

        for name, metric in train_metrics.items():
            if name not in logger_hooks:
                logger.add_hook(partial(jointplot, loss_type=f"{name}"), feature=f"val_{name}", freq=1)
                logger_hooks.add(name)
            logger.update(f"train_{name}", np.mean(metric))
        for name, metric in val_metrics.items():
            logger.update(f"val_{name}", np.mean(metric))

if __name__ == "__main__":
    Fire(main)
