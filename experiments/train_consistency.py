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
from task_configs import tasks, RealityTask
from transfers import functional_transfers
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
	fast=False,
	subset_size=None, early_stopping=float('inf'),
	mode='standard',
	max_epochs=800, **kwargs,
):
	
	early_stopping = 8
	loss_config_percepnet = {
        "paths": {
            "y": [tasks.normal],
            "z^": [tasks.principal_curvature],
            "f(y)": [tasks.normal, tasks.principal_curvature],
        },
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("f(y)", "z^"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256, 
                realities=("test", "ood"), 
                paths=[
                    "y",
                    "z^",
                    "f(y)",
                ]
            ),
        },
    }
	
	# CONFIG
	batch_size = 64
	energy_loss = EnergyLoss(**loss_config_percepnet)

	task_list = [tasks.rgb, tasks.normal, tasks.principal_curvature]
	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		task_list,
		batch_size=batch_size, fast=fast,
		subset_size=subset_size,
	)
	test_set = load_test(task_list)
	ood_set = load_ood(task_list)
	print (train_step, val_step)
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, task_list)
	ood = RealityTask.from_static("ood", ood_set, task_list)

	# GRAPH
	realities = [train, val, test, ood]
	graph = TaskGraph(tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature] + realities, pretrained=False, 
		freeze_list=[functional_transfers.n],
	)
	graph.compile(torch.optim.Adam, lr=4e-4, weight_decay=2e-6, amsgrad=True)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	energy_loss.logger_hooks(logger)
	best_val_loss, stop_idx = float('inf'), 0

	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")

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
		if logger.data["val_mse : f(y) -> z^"][-1] < best_val_loss:
			print ("Better val loss, reset stop_idx: ", stop_idx)
			best_val_loss, stop_idx = logger.data["val_mse : f(y) -> z^"][-1], 0
			energy_loss.plot_paths(graph, logger, realities, prefix="best")
			graph.save(weights_dir=f"{RESULTS_DIR}")

		if stop_idx >= early_stopping:
			print ("Stopping training now")
			break



	early_stopping = 50
	# CONFIG
	energy_loss = get_energy_loss(config="perceptual", mode=mode, **kwargs)

	# GRAPH
	realities = [train, val, test, ood]
	graph = TaskGraph(tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature] + realities, pretrained=False, 
		freeze_list=[functional_transfers.f],
	)
	graph.edge(tasks.normal, tasks.principal_curvature).model.load_weights(f"{RESULTS_DIR}/f.pth")
	graph.compile(torch.optim.Adam, lr=4e-4, weight_decay=2e-6, amsgrad=True)

	# LOGGING
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	energy_loss.logger_hooks(logger)
	best_val_loss, stop_idx = float('inf'), 0

	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")

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
			graph.save(f"{RESULTS_DIR}/graph.pth")

		if stop_idx >= early_stopping:
			print ("Stopping training now")
			break

	



if __name__ == "__main__":
	Fire(main)
