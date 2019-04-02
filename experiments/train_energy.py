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
	pretrained=True, finetuned=True, fast=False, batch_size=None, ood_batch_size=None, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	ood_batch_size = ood_batch_size or batch_size
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		[tasks.rgb, tasks.normal, tasks.principal_curvature],
		return_dataset=True,
		batch_size=batch_size,
		train_buildings=["almena"] if fast else None, 
		val_buildings=["almena"] if fast else None,
		resize=256,
	)
	ood_consistency_dataset, _, _, _ = load_train_val(
		[tasks.rgb,],
		return_dataset=True,
		train_buildings=["almena"] if fast else None, 
		val_buildings=["almena"] if fast else None,
		resize=512,
	)
	# train_subset_dataset = load_all(
	# 	tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature],
	# 	batch_size=batch_size,
	# 	return_dataset=True,
	# 	train_buildings=["almena"] if fast else None, 
	# 	val_buildings=["almena"] if fast else None,
	# 	resize=256,
	# 	subset_size=64,
	# )

	train_step, val_step = train_step//4, val_step//4
	if fast: train_step, val_step = 20, 20
	test_set = load_test([tasks.rgb, tasks.normal, tasks.principal_curvature])
	ood_images = load_ood()
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	ood_consistency = RealityTask("ood_consistency", ood_consistency_dataset, batch_size=ood_batch_size, shuffle=True)
	# train_1perc = RealityTask("train_subset", train_subset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, [tasks.rgb, tasks.normal, tasks.principal_curvature])
	ood = RealityTask.from_static("ood", (ood_images,), [tasks.rgb,])

	energy_loss.load_realities([train, val, ood_consistency, test, ood])

	# GRAPH
	graph = TaskGraph(tasks=energy_loss.tasks + [train, val, ood_consistency, test, ood], finetuned=finetuned)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	graph.load_weights(f"{MODELS_DIR}/conservative/conservative.pth")
	print (graph)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(
		lambda _, __: graph.save(f"{RESULTS_DIR}/graph_{loss_config}.pth"), feature="epoch", freq=1,
	)
	graph.save(f"{RESULTS_DIR}/graph_{loss_config}.pth")
	energy_loss.logger_hooks(logger)

	# TRAINING
	for epochs in range(0, 800):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, prefix="start" if epochs == 1 else "")
		if visualize: return

		graph.train()
		for _ in range(0, train_step):
			train.step()
			train_loss = energy_loss(graph, reality=train)
			graph.step(train_loss)
			logger.update("loss", train_loss)

			# train_1perc.step()
			# train_1perc_loss = energy_loss(graph, reality=train_1perc)
			# graph.step(train_1perc_loss)

			ood_consistency.step()
			ood_consistency_loss = energy_loss(graph, reality=ood_consistency)
			graph.step(ood_consistency_loss)

		graph.eval()
		for _ in range(0, val_step):
			val.step()
			with torch.no_grad():
				val_loss = energy_loss(graph, reality=val)
			logger.update("loss", val_loss)

		energy_loss.logger_update(logger)
		logger.step()

if __name__ == "__main__":
	Fire(main)
