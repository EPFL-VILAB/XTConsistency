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
from scipy.stats import pearsonr

import IPython


def main(
	pretrained=True, finetuned=True, fast=False, batch_size=None, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or 8
	energy_loss = get_energy_loss(config="visualize", mode="standard", **kwargs)

	# DATA LOADING
	train_dataset, val_dataset, train_step, val_step = load_train_val(
		energy_loss.tasks, batch_size=batch_size,
		return_dataset=True,
		train_buildings=["almena"], 
		val_buildings=["almena"],
	)
	# test_set = load_test(energy_loss.tasks)
	# ood_images = load_ood()
	
	train = RealityTask("train", train_dataset, energy_loss.tasks, 
		batch_size=batch_size, shuffle=True
	)
	# test = RealityTask.from_static("test", test_set, energy_loss.tasks)
	# ood = RealityTask.from_static("ood", (ood_images,), [tasks.rgb])

	# GRAPH
	graph = TaskGraph(tasks=energy_loss.tasks + [train], finetuned=finetuned)
	graph.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	print (graph)
	# graph.load_weights(f"{MODELS_DIR}/conservative/graph_rgb_normal_curv.pth")
	
	# LOGGING
	logger = VisdomLogger("train", env=JOB)
	def jointplot(logger, data):
	    data = np.stack((data["energy_loss"], data["mse_loss"]), axis=1)
	    logger.plot(data, "losses", opts={"legend": ["energy_loss", "mse_loss"]})
	logger.add_hook(jointplot, feature='mse_loss', freq=20)


	# let's do a resolution change from 128 to 192 to 256 to 320 to 384 to 448 to 512
	#									0      100.   200.   300    400.   500.   600

	logger.text("0-100 size 128.")
	logger.text("100-200 size 192.")
	logger.text("200-300 size 256.")
	logger.text("300-400 size 320.")
	logger.text("400-500 size 384.")
	logger.text("500-600 size 448.")
	logger.text("600-700 size 512.")

	elosses, mlosses = [], []
	for epochs in range(0, 700):
		train.step()
		print (train.task_data[tasks.rgb_domain_shift].shape)
		train_loss = energy_loss(graph, reality=train)
		eloss = energy_loss.metrics[train]["F(RC(x)) -> n(x)"][-1]
		mloss = energy_loss.metrics[train]["n(x) -> y^"][-1]
		elosses += [eloss.data.cpu().numpy().mean()]
		mlosses += [mloss.data.cpu().numpy().mean()]

		# print ("Finetuned: ", eloss, mloss)

		logger.update("energy_loss", eloss)
		logger.update("mse_loss", mloss)

		if (epochs+1) % 100 == 0:
			train.timestep()

	R, p = pearsonr(np.array(elosses), np.array(mlosses))
	logger.text(f"Correlation {R:0.3f} (p={p:0.5f})")


if __name__ == "__main__":
	Fire(main)
