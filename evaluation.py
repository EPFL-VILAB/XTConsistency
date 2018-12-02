
import numpy as np
import os, sys, math, random, glob, time, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, get_model, tasks
from logger import Logger, VisdomLogger
from datasets import TaskDataset

from modules.unet import UNet, UNetOld

import IPython


### Use case: when finetuning a rgb2normal network
### Inputs rgb2normal/depth net, image-task dataset 

### Depth/normal

### Almena Depth/normal
### Almena corrupted intensity 1 Depth/normal
### Almena corrupted intensity 2 Depth/normal
### Almena corrupted intensity 3 Depth/normal
### Almena corrupted intensity 4 Depth/normal

### Almena PGD epsilon 1e-3 Depth/normal
### Almena PGD epsilon 1e-1 Depth/normal

### Sintel Depth/normal
### NYU Depth/normal


class EvaluationDataset(TaskDataset):

    def __init__(self, src_task=get_task("rgb"), dest_task=get_task("normal")):
        super().__init__(["almena"], tasks=[src_task, dest_task])
        self.src_task, self.dest_task = src_task, dest_task

    def get_metrics(self, pred, target):
        """ Gets standard set of metrics for predictions and targets """

        masks = self.dest_task.build_mask(target, val=self.dest_task.mask_val)
        original_pred, original_target, masks = (x.data.permute((0, 2, 3, 1)).cpu().numpy() for x in [pred, target, masks])
        masks = masks[:, :, :, 0]
        

        norm = lambda a: np.sqrt((a * a).sum(axis=1))
        def cosine_similarity(x1, x2, dim=1, eps=1e-8):
            w12 = np.sum(x1 * x2, dim)
            w1, w2 = norm(x1), norm(x2)
            return (w12 / (w1 * w2).clip(min=eps)).clip(min=-1.0, max=1.0)

        original_pred = original_pred.astype('float64')
        original_target = original_target.astype('float64')
        num_examples, width, height, num_channels = original_pred.shape

        pred = original_pred.reshape([-1, num_channels])
        target = original_target.reshape([-1, num_channels])
        num_valid_pixels, num_invalid_pixels = np.sum(masks), np.sum(1 - masks)

        ang_errors_per_pixel_unraveled = np.arccos(cosine_similarity(pred, target)) * 180 / math.pi
        ang_errors_per_pixel = ang_errors_per_pixel_unraveled.reshape(num_examples, width, height)
        ang_errors_per_pixel_masked = ang_errors_per_pixel * masks
        ang_error_mean = np.sum(ang_errors_per_pixel_masked) / num_valid_pixels
        ang_error_without_masking = np.mean(ang_errors_per_pixel)
        ang_error_median = np.mean(np.median(np.ma.masked_equal(ang_errors_per_pixel_masked, 0), axis=1))

        normed_pred = pred / (norm(pred)[:, None] + 2e-1)
        normed_target = target / (norm(target)[:, None] + 2e-1)
        masks_expanded = np.expand_dims(masks, 3).reshape([-1])
        mse = (normed_pred - normed_target) * masks_expanded[:, None]
        mse, rmse = np.mean(mse ** 2), np.sqrt(np.mean(mse ** 2)) * 255.0

        threshold_1125 = (np.sum(ang_errors_per_pixel_masked <= 11.25) - num_invalid_pixels) / num_valid_pixels
        threshold_225 = (np.sum(ang_errors_per_pixel_masked <= 22.5) - num_invalid_pixels) / num_valid_pixels
        threshold_30 = (np.sum(ang_errors_per_pixel_masked <= 30) - num_invalid_pixels) / num_valid_pixels
        
        return {
            "ang_error_without_masking": ang_error_without_masking,
            "ang_error_mean": ang_error_mean,
            "ang_error_median": ang_error_median,
            "mse": mse,
            "rmse": rmse,
            'percentage_within_11.25_degrees': threshold_1125,
            'percentage_within_22.5_degrees': threshold_225,
            'percentage_within_30_degrees': threshold_30,
        }

    def evaluate(self, model, logger=None):
        """ Evaluates dataset on model. """

        eval_loader = torch.utils.data.DataLoader(
            self, batch_size=4,
            num_workers=4, shuffle=False, pin_memory=True
        )
        images = torch.cat([x for x, y in eval_loader], dim=0)
        preds, targets, _, _ = model.predict_with_data(eval_loader)
        print (images.shape, preds.shape, targets.shape)
        print (self.get_metrics(preds, targets))







if __name__ == "__main__":
    
    model = DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/unet_percepstep_0.1.pth")
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    dataset = EvaluationDataset()
    dataset.evaluate(model)










