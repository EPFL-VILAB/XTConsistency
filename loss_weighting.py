import numpy as np
import random, sys, os, time, glob, math, itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

import IPython

def dot(grad1, grad2):
	return (grad1 * grad2).sum()

def calculate_weight(model, loss1, loss2, normalize=True):

	grad1 = torch.autograd.grad(loss1, model.parameters(), retain_graph=True)
	grad1 = torch.cat([x.view(-1) for x in grad1])

	model.zero_grad()

	grad2 = torch.autograd.grad(loss2, model.parameters(), retain_graph=True)
	grad2 = torch.cat([x.view(-1) for x in grad2])


	if normalize:
		grad1 = grad1 / torch.norm(grad1)
		grad2 = grad2 / torch.norm(grad2)

	v1v1 = dot(grad1, grad1)
	v1v2 = dot(grad1, grad2)
	v2v2 = dot(grad2, grad2)

	if v1v2 >= v1v1:
		c = torch.tensor(1.0, device=loss1.device)
	elif v1v2 >= v2v2:
		c = torch.tensor(0.0, device=loss1.device)
	else:
		# Case when min norm is perpendciular to the line
		c = dot(grad2 - grad1, grad2) / dot(grad1-grad2, grad1-grad2)**0.5
		# c = (-1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) ))
	return c, 1-c
	# This is always not normalized regardless of you normalized first or not (ngrad and grad are different things)
	# final_grad = (1-c)*grad2 + c*grad1