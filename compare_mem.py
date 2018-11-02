import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.resnet import ResNet
from modules.percep_nets import DenseNet, DeepNet, BaseNet, WideNet, PyramidNet, Dense1by1Net, DenseKernelsNet
from modules.depth_nets import UNetDepth
from modules.unet import UNet

from sklearn.model_selection import train_test_split
from fire import Fire

import IPython


def get_max_batch(model, base=32, max_batch=512):
    i = 0
    try:
        for i in range( (max_batch - base) // 8):
            model.forward(torch.randn(base + (8*i), 3, 256, 256))
            print(base + (8*i))
    except:
      pass
    return base + (8*(i-1))

def main(model1_cls, model2_cls=None, base1=32, base2=32, max_batch=512):
    model1 = DataParallelModel(eval(model1_cls + "()").cuda())
    print("done initializing models")
    mem1 = get_max_batch(model1, base1, max_batch)
    print(f'{model1_cls}: {mem1}')
    if model2_cls is not None:
        model2 = DataParallelModel(eval(model2_cls + "()").cuda())
        mem2 = get_max_batch(model2, base2, max_batch)
        print(f'{model2_cls}: {mem2}')
        print(f'mem2 / mem1 = {mem2 / mem1}')

if __name__ == "__main__":
    Fire(main)