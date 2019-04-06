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
	loss_config="conservative_full", mode="standard", visualize=False,
	pretrained=True, finetuned=False, fast=False, batch_size=None, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.get_tasks("train"),
		batch_size=batch_size, fast=fast,
	)
	test_set = load_test(energy_loss.get_tasks("test"))
	ood_set = load_ood(energy_loss.get_tasks("ood"))

	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
	ood = RealityTask.from_static("ood", ood_set, energy_loss.get_tasks("ood"))

	# GRAPH
	realities = [train, val, test, ood]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, finetuned=finetuned)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	graph.load_weights(f"{MODELS_DIR}/conservative/conservative.pth")

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(lambda _, __: graph.save(f"{RESULTS_DIR}/graph.pth"), feature="epoch", freq=1)
	energy_loss.logger_hooks(logger)

	# TRAINING
	for epochs in range(0, 800):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")
		if visualize: return

		graph.train()
		for _ in range(0, train_step):
			train_loss = energy_loss(graph, realities=[train])
			graph.step(train_loss)
			logger.update("loss", train_loss)
			train.step()

		graph.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, realities=[val])
			logger.update("loss", val_loss)
			val.step()

		energy_loss.logger_update(logger)
		logger.step()

if __name__ == "__main__":
	Fire(main)
