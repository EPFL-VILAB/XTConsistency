import os, sys, math, random, itertools, time
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
from transfers import functional_transfers
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
	loss_config="conservative_full", mode="standard", visualize=False,
	fast=False, batch_size=None, path=None,
	subset_size=None, early_stopping=float('inf'),
	max_epochs=800, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.get_tasks("train"),
		batch_size=batch_size, fast=fast,
		subset_size=subset_size,
	)
	test_set = load_test(energy_loss.get_tasks("test"))
	ood_set = load_ood(energy_loss.get_tasks("ood"))
	print (train_step, val_step)
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
	ood = RealityTask.from_static("ood", ood_set, energy_loss.get_tasks("ood"))

	# GRAPH
	realities = [train, val, test, ood]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, 
		freeze_list=energy_loss.freeze_list,
	)
	graph.edge(tasks.rgb, tasks.normal).model = None 
	graph.edge(tasks.rgb, tasks.normal).path = path
	graph.edge(tasks.rgb, tasks.normal).load_model()
	graph.compile(torch.optim.Adam, lr=4e-4, weight_decay=2e-6, amsgrad=True)
	graph.save(weights_dir=f"{RESULTS_DIR}")

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	energy_loss.logger_hooks(logger)
	best_val_loss, stop_idx = float('inf'), 0

	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")
		if visualize: return

		graph.train()
		for _ in range(0, train_step):
			train_loss = energy_loss(graph, realities=[train])
			train_loss = sum([train_loss[loss_name] for loss_name in train_loss])

			graph.step(train_loss)
			train.step()
			logger.update("loss", train_loss)

		graph.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, realities=[val])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
			val.step()
			logger.update("loss", val_loss)

		energy_loss.logger_update(logger)
		logger.step()

		stop_idx += 1
		if logger.data["val_mse : n(x) -> y^"][-1] < best_val_loss:
			print ("Better val loss, reset stop_idx: ", stop_idx)
			best_val_loss, stop_idx = logger.data["val_mse : n(x) -> y^"][-1], 0
			energy_loss.plot_paths(graph, logger, realities, prefix="best")
			graph.save(weights_dir=f"{RESULTS_DIR}")

		if stop_idx >= early_stopping:
			print ("Stopping training now")
			return

if __name__ == "__main__":
	Fire(main)
