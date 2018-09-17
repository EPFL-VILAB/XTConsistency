#utils.py

import numpy as np
import random, sys, os, time, glob, math
import random

JOB = open('jobinfo.txt').read().strip()

try:
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torch.optim as optim
	from torch.autograd import Variable

	DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
except:
	pass

def batch(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield arr
			arr = []
	if len(arr) != 0:
		yield arr

def batched(datagen, batch_size=32):
	arr = []
	for data in datagen:
		arr.append(data)
		if len(arr) == batch_size:
			yield list(zip(*arr))
			arr = []
	if len(arr) != 0:
		yield list(zip(*arr))

def elapsed(times=[time.time()]):
	times.append(time.time())
	return times[-1] - times[-2]

# Cycles through iterable without making extra copies
def cycle(iterable):
	while True:
		for i in iterable:
			yield i

def build_mask(target, val=0.0, tol=1e-3):
	if target.shape[1] == 1:
		return ~((target >= val - tol) & (target <= val + tol))
	
	mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
	mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
	mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
	mask = ~(mask1 & mask2 & mask3).unsqueeze(1).expand_as(target)
	return mask
