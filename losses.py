import numpy as np
import random, sys, os, time, glob, math, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from functools import partial

import IPython


def dot(grad1, grad2):
    return (grad1 * grad2).sum()

def calculate_weight(model, loss1, loss2, normalize=True):

    grad1 = torch.autograd.grad(loss1, model.parameters(), retain_graph=True)
    grad1 = torch.cat([x.view(-1) for x in grad1])

    model.zero_grad()

    grad2 = torch.autograd.grad(loss2, model.parameters(), retain_graph=True)
    grad2 = torch.cat([x.view(-1) for x in grad2])


    if normalize:
        grad1 = grad1 / torch.norm(grad1)
        grad2 = grad2 / torch.norm(grad2)

    v1v1 = dot(grad1, grad1)
    v1v2 = dot(grad1, grad2)
    v2v2 = dot(grad2, grad2)

    if v1v2 >= v1v1:
        c = torch.tensor(1.0, device=loss1.device)
    elif v1v2 >= v2v2:
        c = torch.tensor(0.0, device=loss1.device)
    else:
        # Case when min norm is perpendciular to the line
        c = dot(grad2 - grad1, grad2) / dot(grad1-grad2, grad1-grad2)**0.5
        # c = (-1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) ))
    return c, 1-c
    # This is always not normalized regardless of you normalized first or not (ngrad and grad are different things)
    # final_grad = (1-c)*grad2 + c*grad1


def get_standardization_mixed_loss_fn(curvature_model, depth_model, logger, include_depth, standardization_window_size):
    return partial(mixed_loss,
                   curvature_model=curvature_model,
                   depth_model=depth_model,
                   logger=logger,
                   include_depth=include_depth,
                   standardization_window_size=standardization_window_size,
                   )

def get_angular_distances(pred, target):
    num_channels = pred.shape[-1]

    output = cosine_similarity(pred.view([-1, num_channels]), target.view([-1, num_channels]))
    return torch.acos(output) * 180 / math.pi

def mixed_loss(pred, target, curvature_model, depth_model, logger, include_depth, standardization_window_size):
    mask = build_mask(target.detach(), val=0.502)
    mse = F.mse_loss(pred * mask.float(), target * mask.float())
    curvature = F.mse_loss(curvature_model(pred) * mask.float(), curvature_model(target) * mask.float())
    depth = F.mse_loss(depth_model(pred) * mask.float(), depth_model(target) * mask.float())

    if "train_mse_loss" in logger.data and len(logger.data["train_mse_loss"]) >= 2:
        normals_loss_std = np.std(logger.data["train_mse_loss"][-standardization_window_size:])
        curvature_loss_std = np.std(logger.data["train_curvature_loss"][-standardization_window_size:])
        depth_loss_std = np.std(logger.data["train_curvature_loss"][-standardization_window_size:])
        final_loss = mse / float(normals_loss_std)
        final_loss += curvature / float(curvature_loss_std)
        if include_depth:
            final_loss += depth / float(depth_loss_std)
    else:
        final_loss = mse + curvature

    metrics_to_return = (mse.detach(), curvature.detach(), depth.detach())
    return final_loss, metrics_to_return