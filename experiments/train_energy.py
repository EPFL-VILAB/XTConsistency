import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph
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
	loss_config="conservative_full", mode="standard",
	pretrained=True, finetuned=False, fast=False, batch_size=None, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_loader, val_loader, train_step, val_step = load_train_val(
		energy_loss.tasks, batch_size=batch_size,
		train_buildings=["almena"] if fast else None, 
		val_buildings=["almena"] if fast else None,
	)
	if fast: train_step, val_step = 20, 20
	test_set = load_test(energy_loss.tasks)
	ood_images = load_ood()
	
	train = RealityTask.from_dataloader("train", train_loader, energy_loss.tasks)
	val = RealityTask.from_dataloader("val", val_loader, energy_loss.tasks)
	test = RealityTask.from_static("test", test_set, energy_loss.tasks)
	ood = RealityTask.from_static("ood", (ood_images,), [tasks.rgb])

	# GRAPH
	graph = TaskGraph(tasks=energy_loss.tasks + [train, val, test, ood], finetuned=finetuned)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	print (graph.edge_map)
	IPython.embed()

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(
		lambda logger, data: graph.save(f"{RESULTS_DIR}/graph_{loss_config}.pth"),
		feature="epoch", freq=1,
	)
	energy_loss.logger_hooks(logger)

	# TRAINING
	for epochs in range(0, 800):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, reality=test)
		energy_loss.plot_paths(graph, logger, reality=ood)
		logger.text ("Plotted paths")

		for _ in range(0, train_step):
			train.step()
			train_loss = energy_loss(graph, reality=train)
			graph.step(train_loss)
			logger.update("loss", train_loss)

		for _ in range(0, val_step):
			val.step()
			with torch.no_grad():
				val_loss = energy_loss(graph, reality=val)
			logger.update("loss", val_loss)
		
		energy_loss.logger_update(logger, reality=train)
		energy_loss.logger_update(logger, reality=val)
		logger.step()


if __name__ == "__main__":
	Fire(main)
