
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os, sys, math, random, tarfile, glob, time
import parse

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

	def __init__(self, buildings, source_task='rgb', dest_task='normal'):
		"""
		Args:
			csv_file (string): Path to the csv file with annotations.
			root_dir (string): Directory with all the images.
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.source_task, self.dest_task, self.buildings = source_task, dest_task, buildings
		self.source_files = [sorted(glob.glob(f"/data/{building}_{source_task}/{source_task}/*.png"))
							 for building in buildings]
		self.source_files = [y for x in self.source_files for y in x]
		self.source_transforms = transforms.Compose([
									transforms.ToTensor()])
		self.dest_transforms = transforms.Compose([
									transforms.ToTensor()])

	def __len__(self):
		return len(self.source_files)

	def __getitem__(self, idx):
		
		source_file = self.source_files[idx]

		result = parse.parse("/data/{building}_{task}/{task}/{view}_domain_{task}.png", source_file)
		building, task, view = result["building"], result["task"], result["view"]
		dest_file = f"/data/{building}_{self.dest_task}/{self.dest_task}/{view}_domain_{self.dest_task}.png"

		try:
			image = Image.open(source_file)
			print (image.shape)
			image = self.source_transforms(image)
			print (image.dtype)

			task = Image.open(dest_file)
			print (task.shape)
			task = self.dest_transforms(task)
			print (source_file, task.dtype)
			return image, task
		except:
			#print ("Error in file pair: ", source_file, dest_file)
			time.sleep(0.1)
			return self.__getitem__(random.randrange(0, len(self.source_files)))

		


if __name__ == "__main__":

	dataset = ImageTaskDataset(buildings=["ackermanville", "adairsville", "adrian", "airport", "akiak"])
	data_loader = torch.utils.data.DataLoader(dataset, num_workers=32, batch_size=32, shuffle=True)
	logger = Logger("data")
	logger.add_hook(lambda data: logger.step(), freq=16)

	for i, (X, Y) in enumerate(data_loader):
		logger.update('epoch', i)
