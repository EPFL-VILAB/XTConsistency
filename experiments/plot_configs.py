import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph, Discriminator
from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
	loss_configs=["rgb2x_plots", "rgb2x_plots_size320", "rgb2x_plots_size320", "rgb2x_plots_size384", "rgb2x_plots_size448", "rgb2x_plots_size512"], mode="standard", visualize=False,
	pretrained=True, finetuned=False, batch_size=None, 
	**kwargs,
):
	
	logger = VisdomLogger("train", env=JOB)
	for loss_config in loss_configs:
	
		# CONFIG
		batch_size = batch_size or 32
		energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

		# DATA LOADING
		test_set = load_test(energy_loss.get_tasks("test"))
		ood_set = load_ood(energy_loss.get_tasks("ood"))
		
		test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
		ood = RealityTask.from_static("ood", ood_set, energy_loss.get_tasks("ood"))

		# GRAPH
		realities = [test, ood]
		graph = TaskGraph(tasks=energy_loss.tasks + realities, finetuned=finetuned)
		graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

		# LOGGING
		energy_loss.plot_paths(graph, logger, realities, prefix=loss_config)

if __name__ == "__main__":
	Fire(main)
