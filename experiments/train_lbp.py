import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss, EnergyLoss
from graph import TaskGraph, Discriminator
from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_train_val, load_test, load_ood
from transfers import get_transfer_name, Transfer
from task_configs import tasks, RealityTask
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
	pretrained=True, finetuned=False, fast=False, batch_size=None, 
	cont=f"{MODELS_DIR}/conservative/conservative.pth", 
	max_epochs=800, **kwargs,
):
	
	task_list = [
		tasks.rgb,
		tasks.normal,
		tasks.principal_curvature,
		tasks.depth_zbuffer,
		# tasks.sobel_edges,
	]
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		task_list,
		batch_size=batch_size, fast=fast,
	)
	test_set = load_test(task_list)
	train_step, val_step = train_step//4, val_step//4
	print (train_step, val_step)
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, task_list)
	# ood = RealityTask.from_static("ood", ood_set, [tasks.rgb,])

	# GRAPH
	realities = [train, val, test]
	graph = TaskGraph(tasks=task_list + realities, finetuned=finetuned)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(lambda _, __: graph.save(f"{RESULTS_DIR}/graph.pth"), feature="epoch", freq=1)
	logger.add_hook(partial(jointplot, loss_type=f"energy"), feature=f"val_energy", freq=1)

	def path_name(path):
		if len(path) == 1:
			print (path)
			return str(path[0])
		if str((path[-2].name, path[-1].name)) not in graph.edge_map: return None
		sub_name = path_name(path[:-1])
		if sub_name is None: return None
		return f'{get_transfer_name(Transfer(path[-2], path[-1]))}({sub_name})'
	
	for i in range(0, 20):

		gt_paths = {path_name(path): list(path) for path in itertools.permutations(task_list, 1) if path_name(path) is not None}
		baseline_paths = {path_name(path): list(path) for path in itertools.permutations(task_list, 2) if path_name(path) is not None}
		paths = {path_name(path): list(path) for path in itertools.permutations(task_list, 3) if path_name(path) is not None}
		selected_paths = dict(random.sample(paths.items(), k=3))

		print ("Chosen paths: ", selected_paths)

		loss_config = {
			"paths": {**gt_paths, **baseline_paths, **paths},
			"losses": {
				"baseline_mse": {
					("train", "val",): [
					   (path_name, str(path[-1])) for path_name, path in baseline_paths.items()
					]
				},
				"mse": {
					("train", "val",): [
					   (path_name, str(path[-1])) for path_name, path in selected_paths.items()
					]
				},
				"eval_mse": {
					("val",): [
					   (path_name, str(path[-1])) for path_name, path in paths.items()
					]
				}
			},
			"plots": {
				"ID": dict(
					size=256, 
					realities=("test",), 
					paths=[
						path_name for path_name, path in selected_paths.items()
					] + [
						str(path[-1]) for path_name, path in selected_paths.items()
					]
				),
			},
		}
		
		energy_loss = EnergyLoss(**loss_config)
		energy_loss.logger_hooks(logger)

		# TRAINING
		for epochs in range(0, 5):

			logger.update("epoch", epochs)
			energy_loss.plot_paths(graph, logger, realities, prefix="")

			graph.train()
			for _ in range(0, train_step):
				train_loss = energy_loss(graph, realities=[train], loss_types=["mse", "baseline_mse"])
				train_loss = sum([train_loss[loss_name] for loss_name in train_loss])

				graph.step(train_loss)
				train.step()
				logger.update("loss", train_loss)

			graph.eval()
			for _ in range(0, val_step):
				with torch.no_grad():
					val_loss = energy_loss(graph, realities=[val], loss_types=["mse", "baseline_mse"])
					val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
				val.step()
				logger.update("loss", val_loss)

			val_loss = energy_loss(graph, realities=[val], loss_types=["eval_mse"])
			val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
			val.step()

			logger.update("train_energy", val_loss)
			logger.update("val_energy", val_loss)

			energy_loss.logger_update(logger)
			logger.step()

if __name__ == "__main__":
	Fire(main)
