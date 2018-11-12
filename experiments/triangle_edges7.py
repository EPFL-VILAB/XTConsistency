
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
from transfers import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from fire import Fire

from skimage import feature
from functools import partial
from scipy import ndimage

import IPython


def main(curvature_step=0, depth_step=0):

    curvature_weight = curvature_step
    depth_weight = depth_step

    # MODEL
    print ("Using UNet")
    model = DataParallelModel(UNet())
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    scheduler = MultiStepLR(model.optimizer, milestones=[5*i + 1 for i in range(0, 80)], gamma=0.95)

    def mixed_loss(pred, target):
        mask = build_mask(target.detach(), val=0.502)
        F, G, f, s, H_g, h_f = curvature2normal, depth2normal, curvature_model, normal2edge, curve_cycle, depth_cycle
        CE, EC = curvature2edges, edges2curvature
        ax, y, y_hat = target, pred, target
        mse_loss = lambda x, y: ((x*mask.float() -y*mask.float())**2).mean()

        cycle = mse_loss(F(f(y)), y)
        curve_edge = mse_loss(ax, CE(f(y)))
        inversecurve_edge = mse_loss(EC(ax), f(y))

        c1, c2, c3 = 1/cycle.detach(), 1/curve_edge.detach(), 1/inversecurve_edge.detach()
        
        # curvature = torch.tensor(0.0, device=mse.device) if curvature_weight == 0.0 else \
        #     F.mse_loss(curvature_model(pred) * mask.float(), (target) * mask.float())
        # depth = torch.tensor(0.0, device=mse.device) if depth_weight == 0.0 else \
        #     F.mse_loss(depth_model(pred) * mask.float(), depth_model(target) * mask.float())

        return c1*cycle + c2*curve_edge + c3*inversecurve_edge, (cycle.detach(), curve_edge.detach(), inversecurve_edge.detach())

    # LOGGING
    logger = VisdomLogger("train", env=JOB)
    logger.add_hook(lambda x: logger.step(), feature="loss", freq=10)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="cycle_loss"), feature="val_cycle_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="curve_edge_loss"), feature="val_curve_edge_loss", freq=1)
    logger.add_hook(partial(jointplot, logger=logger, loss_type="inversecurve_edge_loss"), feature="val_inversecurve_edge_loss", freq=1)
    logger.add_hook(lambda x: model.save(f"{RESULTS_DIR}/model.pth"), feature="loss", freq=400)

    def dest_transforms(x):
        image = x.data.cpu().numpy().mean(axis=0)
        blur = ndimage.filters.gaussian_filter(image, sigma=2)
        sx = ndimage.sobel(blur, axis=0, mode='constant')
        sy = ndimage.sobel(blur, axis=1, mode='constant')
        image = np.hypot(sx, sy)
        edge = torch.FloatTensor(image).unsqueeze(0)
        return edge

    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "rgb", batch_size=32, dest_transforms=dest_transforms)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=-1.0, 
        loss_models={
            "f(y) curve": curvature_model, 
            "F(f(y)) cycle": lambda x: curvature2normal(curvature_model(x)),
            "s(y) edges": normal2edge,
            "CE(f(y)) curvature edges": lambda x: curvature2edges(curvature_model(x)),
        },
        loss_targets=False
    )
    plot_images(model, logger, test_set, mask_val=-1.0, 
        loss_models={
            "EC(a(x)) inverse curvature edges": lambda x: edges2curvature(x),
        },
        loss_preds=False,
    )

    # TRAINING
    for epochs in range(0, 800):
        logger.update("epoch", epochs)

        train_set = itertools.islice(train_loader, train_step)
        (cycle_data, curve_edge_data, inversecurve_edge_data) = model.fit_with_metrics(
            train_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("train_cycle_loss", np.mean(cycle_data))
        logger.update("train_curve_edge_loss", np.mean(curve_edge_data))
        logger.update("train_inversecurve_edge_loss", np.mean(inversecurve_edge_data))

        val_set = itertools.islice(val_loader, val_step)
        (cycle_data, curve_edge_data, inversecurve_edge_data) = model.predict_with_metrics(
            val_set, loss_fn=mixed_loss, logger=logger
        )
        logger.update("val_cycle_loss", np.mean(cycle_data))
        logger.update("val_curve_edge_loss", np.mean(curve_edge_data))
        logger.update("val_inversecurve_edge_loss", np.mean(inversecurve_edge_data))

        plot_images(model, logger, test_set, ood_images, mask_val=-1.0, 
            loss_models={
                "f(y) curve": curvature_model, 
                "F(f(y)) cycle": lambda x: curvature2normal(curvature_model(x)),
                "s(y) edges": normal2edge,
                "CE(f(y)) curvature edges": lambda x: curvature2edges(curvature_model(x)),
            },
            loss_targets=False
        )
        plot_images(model, logger, test_set, mask_val=-1.0, 
            loss_models={
                "EC(a(x)) inverse curvature edges": lambda x: edges2curvature(x),
            },
            loss_preds=False,
        )
        scheduler.step()


if __name__ == "__main__":
    Fire(main)
