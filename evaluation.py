
import numpy as np
import os, sys, math, random, glob, time, itertools
from fire import Fire

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, DataParallelModel
from task_configs import get_task, get_model, tasks
from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_ood
import transforms

from modules.unet import UNet, UNetOld

import IPython


### Use case: when finetuning a rgb2normal network
### Inputs rgb2normal/depth net, image-task dataset 

### Depth/normal

### Almena Depth/normal Check
### Almena corrupted intensity 1 Depth/normal Check
### Almena corrupted intensity 2 Depth/normal Check
### Almena corrupted intensity 3 Depth/normal Check
### Almena corrupted intensity 4 Depth/normal Check

### Almena PGD epsilon 1e-3 Depth/normal
### Almena PGD epsilon 1e-1 Depth/normal

### Sintel Depth/normal
### NYU Depth/normal


class ValidationMetrics(object):

    PLOT_METRICS = ["ang_error_median", "eval_mse"]

    def __init__(self, name, src_task=get_task("rgb"), dest_task=get_task("normal")):
        self.name = name
        self.src_task, self.dest_task = src_task, dest_task
        self.load_dataset()

    def load_dataset(self):
        self.dataset = TaskDataset(["almena", "albertville"], tasks=[self.src_task, self.dest_task])

    def build_dataloader(self, sample=None, batch_size=64):
        sampler = torch.utils.data.SequentialSampler() if sample is None else \
            torch.utils.data.SubsetRandomSampler(random.Random(229).sample(range(len(self.dataset)), sample))

        eval_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size,
            num_workers=16, sampler=sampler, pin_memory=True
        )
        return eval_loader

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
            "eval_mse": mse,
            "eval_rmse": rmse,
            'percentage_within_11.25_degrees': threshold_1125,
            'percentage_within_22.5_degrees': threshold_225,
            'percentage_within_30_degrees': threshold_30,
        }

    @staticmethod
    def plot(logger):
        for metric in ValidationMetrics.PLOT_METRICS:
            keys = [key for key in logger.data if metric in key]
            data = np.stack((logger.data[key] for key in keys), axis=1)
            logger.plot(data, metric, opts={"legend": keys})

    def evaluate(self, model, logger=None, sample=None, show_images=False):
        """ Evaluates dataset on model. """

        eval_loader = self.build_dataloader(sample=sample)
        images, preds, targets, _, _ = model.predict_with_data(eval_loader)
        metrics = self.get_metrics(preds, targets)

        if logger is not None:
            for metric in ValidationMetrics.PLOT_METRICS:
                logger.update(f"{self.name}_{metric}", metrics[metric])
            if show_images:
                logger.images_grouped([images, preds, targets], self.name, resize=256)

        return metrics




class ImageCorruptionMetrics(ValidationMetrics):

    TRANSFORMS = [
        transforms.resize, 
        transforms.resize_rect, 
        transforms.color_jitter, 
        transforms.scale,
        transforms.rotate,
        transforms.elastic,
        transforms.translate,
        transforms.gauss,
        transforms.motion_blur,
        transforms.noise,
        transforms.flip,
        transforms.impulse_noise,
        transforms.crop,
        transforms.jpeg_transform,
        transforms.brightness,
        transforms.contrast,
        transforms.blur,
        transforms.pixilate,
    ]

    def __init__(self, *args, **kwargs):
        self.corruption = kwargs.pop('corruption', 1)
        super().__init__(*args, **kwargs)

    def build_dataloader(self, sample=None):
        eval_loader = super().build_dataloader(sample=sample, batch_size=8)

        for i, (X, Y) in enumerate(eval_loader):
            transform = self.TRANSFORMS[i % len(self.TRANSFORMS)]
            if transform.geometric:
                images = torch.stack([X, Y], dim=1)
                B, N, C, H, W = images.shape
                result = transform(images.view(B*N, C, H, W).to(DEVICE))
                result = result.view(B, N, C, result.shape[2], result.shape[3])                
                yield result[:, 0], result[:, 1]
            else:
                yield transform(X.to(DEVICE)), Y.to(DEVICE)


class AdversarialMetrics(ValidationMetrics):

    def __init__(self, *args, **kwargs):
        self.model = kwargs.pop('model', None)
        self.eps = kwargs.pop('eps', 1e-3)
        self.n = kwargs.pop('n', 10)
        self.lr = kwargs.pop('lr', 0.9)
        super().__init__(*args, **kwargs)

    def build_dataloader(self, sample=None):
        eval_loader = super().build_dataloader(sample=sample, batch_size=32)

        for i, (X, Y) in enumerate(eval_loader):

            perturbation = torch.randn_like(X).to(DEVICE).requires_grad_(True)
            optimizer = torch.optim.Adam([perturbation], lr=self.lr)

            for i in range(0, self.n):
                perturbation_zc = (perturbation - perturbation.mean())/perturbation.std()
                Xn = (X.to(DEVICE) + perturbation_zc).clamp(min=0, max=1)
                loss, _ = self.dest_task.norm(self.model(Xn), Y.to(DEVICE))
                loss = -1.0*loss
                # print ("Loss: ", loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            with torch.no_grad():
                perturbation_zc = (perturbation - perturbation.mean())/perturbation.std()
                Xn = (X + self.eps*perturbation_zc.cpu()).clamp(min=0, max=1)

            yield Xn.detach(), Y.detach()

    def evaluate(self, model, logger=None, sample=None, show_images=False):
        self.model = model #set the adversarial model to the current model
        return super().evaluate(model, logger=logger, sample=sample, show_images=show_images)

datasets = [
    ValidationMetrics("almena"),
    # ImageCorruptionMetrics("almena_corrupted1", corruption=1),
    # ImageCorruptionMetrics("almena_corrupted2", corruption=2),
    # ImageCorruptionMetrics("almena_corrupted3", corruption=3),
    ImageCorruptionMetrics("almena_corrupted4", corruption=4),
    # AdversarialMetrics("almena_adversarial_eps0.005", eps=5e-3),
    AdversarialMetrics("almena_adversarial_eps0.01", eps=1e-2),
]

def run_eval_suite(model=None, logger=None, model_file="unet_percepstep_0.1.pth", sample=80, show_images=False):
    load_ood()
    model = model or DataParallelModel.load(UNetOld().cuda(), f"{MODELS_DIR}/{model_file}")
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)

    logger = logger or VisdomLogger("eval", env=JOB)

    for dataset in datasets:
        print (dataset.evaluate(model, sample=sample, logger=logger, show_images=show_images))

    ValidationMetrics.plot(logger)



if __name__ == "__main__":
    Fire(run_eval_suite)










