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
from transfers import functional_transfers
from evaluation import run_eval_suite

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython

def main(
	loss_config="conservative_full", mode="standard", visualize=False,
	fast=False, batch_size=None, 
	max_epochs=800, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 64)
	energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.get_tasks("train"),
		batch_size=batch_size, fast=fast,
	)
	train_step, val_step = train_step//4, val_step//4
	test_set = load_test(energy_loss.get_tasks("test"))
	ood_set = load_ood(energy_loss.get_tasks("ood"))
	print ("Train step: ", train_step, "Val step: ", val_step)
	
	train = RealityTask("train", train_dataset, batch_size=batch_size, shuffle=True)
	val = RealityTask("val", val_dataset, batch_size=batch_size, shuffle=True)
	test = RealityTask.from_static("test", test_set, energy_loss.get_tasks("test"))
	ood = RealityTask.from_static("ood", ood_set, energy_loss.get_tasks("ood"))

	# GRAPH
	realities = [train, val, test, ood]
	graph = TaskGraph(tasks=energy_loss.tasks + realities, pretrained=True, finetuned=True, 
		freeze_list=[functional_transfers.a, functional_transfers.RC],
	)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(lambda _, __: graph.save(f"{RESULTS_DIR}/graph.pth"), feature="epoch", freq=1)
	energy_loss.logger_hooks(logger)
	
	activated_triangles = set()
	triangle_energy = {"triangle1_mse": float('inf'), "triangle2_mse": float('inf')}

	logger.add_hook(partial(jointplot, loss_type=f"energy"), feature=f"val_energy", freq=1)

	# TRAINING
	for epochs in range(0, max_epochs):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")
		if visualize: return

		graph.train()
		for _ in range(0, train_step):
			# loss_type = random.choice(["triangle1_mse", "triangle2_mse"])
			loss_type = max(triangle_energy, key=triangle_energy.get)

			activated_triangles.add(loss_type)
			train_loss = energy_loss(graph, realities=[train], loss_types=[loss_type])
			train_loss = sum([train_loss[loss_name] for loss_name in train_loss])

			graph.step(train_loss)
			train.step()

			if loss_type == "triangle1_mse":
				consistency_tr1 = energy_loss.metrics["train"]["triangle1_mse : F(RC(x)) -> n(x)"][-1]
				error_tr1 = energy_loss.metrics["train"]["triangle1_mse : n(x) -> y^"][-1]
				triangle_energy["triangle1_mse"] = float(consistency_tr1 / error_tr1)

			elif loss_type == "triangle2_mse":
				consistency_tr2 = energy_loss.metrics["train"]["triangle2_mse : S(a(x)) -> n(x)"][-1]
				error_tr2 = energy_loss.metrics["train"]["triangle2_mse : n(x) -> y^"][-1]
				triangle_energy["triangle2_mse"] = float(consistency_tr2 / error_tr2)
			
			print ("Triangle energy: ", triangle_energy)
			logger.update("loss", train_loss)

			energy = sum(triangle_energy.values())
			if (energy < float('inf')):
				logger.update("train_energy", energy)
				logger.update("val_energy", energy)

		graph.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, realities=[val], loss_types=list(activated_triangles))
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])

			val.step()
			logger.update("loss", val_loss)

		activated_triangles = set()
		energy_loss.logger_update(logger)
		logger.step()

if __name__ == "__main__":
	Fire(main)
