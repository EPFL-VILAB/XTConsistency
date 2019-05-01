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

from functools import partial
from fire import Fire

import IPython
import pdb

def main(
	loss_config="conservative_full", mode="standard", visualize=False,
	pretrained=True, finetuned=False, fast=False, batch_size=None, 
	cont=f"{MODELS_DIR}/conservative/conservative.pth", 
	cont_gan=None, pre_gan=None, **kwargs,
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
	if not USE_RAID: graph.load_weights(cont)
	pre_gan = pre_gan or 1
	discriminator = Discriminator(energy_loss.losses['gan'])
	if cont_gan is not None: discriminator.load_weights(cont_gan)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	logger.add_hook(lambda logger, data: logger.step(), feature="loss", freq=20)
	logger.add_hook(lambda _, __: graph.save(f"{RESULTS_DIR}/graph.pth"), feature="epoch", freq=1)
	logger.add_hook(lambda _, __: discriminator.save(f"{RESULTS_DIR}/discriminator.pth"), feature="epoch", freq=1)
	energy_loss.logger_hooks(logger)

	# TRAINING
	for epochs in range(0, 80):

		logger.update("epoch", epochs)
		energy_loss.plot_paths(graph, logger, realities, prefix="start" if epochs == 0 else "")
		if visualize: return

		graph.train()
		discriminator.train()

		for _ in range(0, train_step):
			if epochs > pre_gan:
				energy_loss.train_iter += 1
				train_loss = energy_loss(graph, discriminator=discriminator, realities=[train])
				train_loss = sum([train_loss[loss_name] for loss_name in train_loss])
				graph.step(train_loss)
				train.step()
				logger.update("loss", train_loss)

			for i in range(5):
				train_loss2 = energy_loss(graph, discriminator=discriminator, realities=[train])
				discriminator.step(train_loss2)
				train.step()
			

		graph.eval()
		discriminator.eval()
		for _ in range(0, val_step):
			with torch.no_grad():
				val_loss = energy_loss(graph, discriminator=discriminator, realities=[val])
				val_loss = sum([val_loss[loss_name] for loss_name in val_loss])
			val.step()
			logger.update("loss", val_loss)

		energy_loss.logger_update(logger)
		logger.step()

if __name__ == "__main__":
	Fire(main)
