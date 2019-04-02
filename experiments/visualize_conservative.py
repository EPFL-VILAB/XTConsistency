import os, sys, math, random, itertools, pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph
from logger import Logger, VisdomLogger
from datasets import TaskDataset, ImageDataset, load_train_val, load_test, load_ood
from task_configs import tasks, RealityTask
from evaluation import run_eval_suite
from transforms import resize

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire
from scipy.stats import pearsonr

import IPython


def main(
	fast=False, batch_size=None, **kwargs,
):
	
	# CONFIG
	batch_size = batch_size or (4 if fast else 32)
	energy_loss = get_energy_loss(config="consistency_two_path", mode="standard", **kwargs)

	# LOGGING
	logger = VisdomLogger("train", env=JOB)

	# DATA LOADING
	video_dataset = ImageDataset(
		files=sorted(
			glob.glob(f"mount/taskonomy_house_tour/original/image*.png"),
			key=lambda x: int(os.path.basename(x)[5:-4])
		),
		return_tuple=True, 
		resize=720,
	)
	video = RealityTask("video", video_dataset, [tasks.rgb,], 
		batch_size=batch_size, shuffle=False
	)

	# GRAPHS
	graph_baseline = TaskGraph(tasks=energy_loss.tasks + [video], finetuned=False)
	graph_baseline.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

	graph_finetuned = TaskGraph(tasks=energy_loss.tasks + [video], finetuned=True)
	graph_finetuned.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)

	graph_conservative = TaskGraph(tasks=energy_loss.tasks + [video], finetuned=True)
	graph_conservative.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	graph_conservative.load_weights(f"{MODELS_DIR}/conservative/conservative.pth")

	graph_ood_conservative = TaskGraph(tasks=energy_loss.tasks + [video], finetuned=True)
	graph_ood_conservative.compile(torch.optim.Adam, lr=3e-5, weight_decay=2e-6, amsgrad=True)
	graph_ood_conservative.load_weights(f"{SHARED_DIR}/results_2F_grounded_1percent_gt_twopath_512_256_crop_7/graph_grounded_1percent_gt_twopath.pth")
	
	graphs = {
		"baseline": graph_baseline, 
		"finetuned": graph_finetuned, 
		"conservative": graph_conservative, 
		"ood_conservative": graph_ood_conservative, 
	}

	inv_transform = transforms.ToPILImage()
	data = {key: {"losses": [], "zooms": []} for key in graphs}
	size = 256
	for batch in range(0, 700):
		
		if batch*batch_size > len(video_dataset.files): break
		
		frac = (batch*batch_size*1.0)/len(video_dataset.files)
		if frac < 0.3:
			size = int(256.0 - 128*frac/0.3)
		elif frac < 0.5:
			size = int(128.0 + 128*(frac-0.3)/0.2)
		else:
			size = int(256.0 + (720 - 256)*(frac-0.5)/0.5)
		print (size)
		# video.reload()
		size = (size//32)*32
		print (size)
		video.step()
		video.task_data[tasks.rgb] = resize(video.task_data[tasks.rgb].to(DEVICE), size).data
		print (video.task_data[tasks.rgb].shape)

		with torch.no_grad():

			for i, img in enumerate(video.task_data[tasks.rgb]):
					inv_transform(img.clamp(min=0, max=1.0).data.cpu()).save(
						f"mount/taskonomy_house_tour/distorted/image{batch*batch_size + i}.png"
					)
			
			for name, graph in graphs.items():
				normals = graph.sample_path([tasks.rgb, tasks.normal], reality=video)
				normals2 = graph.sample_path([tasks.rgb, tasks.principal_curvature, tasks.normal], reality=video)
				
				for i, img in enumerate(normals):
					energy, _ = tasks.normal.norm(normals[i:(i+1)], normals2[i:(i+1)])
					data[name]["losses"] += [energy.data.cpu().numpy().mean()]
					data[name]["zooms"] += [size]
					inv_transform(img.clamp(min=0, max=1.0).data.cpu()).save(
						f"mount/taskonomy_house_tour/normals_{name}/image{batch*batch_size + i}.png"
					)

				for i, img in enumerate(normals2):
					inv_transform(img.clamp(min=0, max=1.0).data.cpu()).save(
						f"mount/taskonomy_house_tour/path2_{name}/image{batch*batch_size + i}.png"
					)

	pickle.dump(data, open(f"mount/taskonomy_house_tour/data.pkl", 'wb'))
	os.system("bash ~/scaling/scripts/create_vids.sh")


if __name__ == "__main__":
	Fire(main)
