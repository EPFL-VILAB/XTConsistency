
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, sys, math, random, tarfile, glob, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from logger import Logger

from PIL import Image
from io import BytesIO
import IPython


class ImageTaskDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, buildings, task='normal'):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.task, self.buildings = task, buildings
		self.image_files = [sorted(glob.glob(f"/data/{building}_rgb/rgb/*.png"))
							 for building in buildings]
		self.task_files = [sorted(glob.glob(f"/data/{building}_{task}/{task}/*.png"))
							 for building in buildings]
		self.image_files = [y for x in self.image_files for y in x]
		self.task_files = [y for x in self.task_files for y in x]
		self.image_transforms = transforms.Compose([
									transforms.ToTensor()])
		self.task_transforms = transforms.Compose([
									transforms.ToTensor()])

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		
		try:
			print ("Image-task pair: ", self.image_files[idx], self.task_files[idx], flush=True)
			image = Image.open(self.image_files[idx])
			image = self.image_transforms(image)

			task = Image.open(self.task_files[idx])
			task = self.task_transforms(task)
			return image, task
		
		except:
			return self.__getitem__(random.randrange(0, len(self.image_files)))

		


if __name__ == "__main__":

	dataset = ImageTaskDataset(buildings=["ackermanville", "adairsville", "adrian", "airport", "akiak"])
	data_loader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=32, shuffle=True)
	logger = Logger("data")
	logger.add_hook(lambda data: logger.step(), freq=16)

	for i, (X, Y) in enumerate(data_loader):
		logger.update('epoch', i)
