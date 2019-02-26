
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *

from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from task_configs import get_task, get_model, tasks
from transfers import functional_transfers
from datasets import load_train_val, load_test, load_ood

from fire import Fire
import IPython
from torch.utils.checkpoint import checkpoint

x = torch.randn(2, 3, 256, 256).requires_grad_()

functional_transfers.n.checkpoint = True
functional_transfers.RC.checkpoint = True

y = functional_transfers.n(x) 
z = functional_transfers.RC(x)

loss = y.mean() + z.mean()
loss.backward()

print (loss)

