
import os, sys, random, yaml
from itertools import product
from tqdm import tqdm

import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

from utils import *
import IPython


""" Model that implements batchwise training with "compilation" and custom loss.
Exposed methods: predict_on_batch(), fit_on_batch(),
Overridable methods: loss(), forward().
"""

class AbstractModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.compiled = False

    # Compile module and assign optimizer + params
    def compile(self, optimizer=None, **kwargs):
        
        if optimizer is not None:
            self.optimizer_class = optimizer
            self.optimizer_kwargs = kwargs

        self.optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        self.compiled = True
        self.to(DEVICE)

    # Predict scores from a batch of data
    def predict_on_batch(self, data):
        self.eval()
        with torch.no_grad():
            return self.forward(data)

    # Fit (make one optimizer step) on a batch of data
    def fit_on_batch(self, data, target, train=True):
        
        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()

        pred = self.forward(data)
        loss = self.loss(pred, target.to(pred.device))

        if train:
            loss.backward()
            self.optimizer.step()

        return pred, float(loss)

    @classmethod
    def load(cls, weights_file=None):
        model = cls()
        if weights_file is not None:
            model.load_state_dict(torch.load(weights_file))
        return model

    def save(self, weights_file):
        torch.save(self.state_dict(), weights_file)

    # Subclasses: override for custom loss + forward functions
    def loss(self, pred, target):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()



""" Model that implements training and prediction on generator objects, with
the ability to print train and validation metrics.
"""

class TrainableModel(AbstractModel):

    def __init__(self):
        super().__init__()

    

    # Fit on generator for one epoch
    def _process_data(self, datagen, train=True, logger=None):

        self.train(train)
        out = []
        for batch, y in datagen:
            y_pred, loss = self.fit_on_batch(batch, y, train=train)
            if logger is not None: logger.update('loss', loss)
            yield ((y_pred.data, y.data, loss))

    def fit(self, datagen, logger=None):
        for x in self._process_data(datagen, train=train, logger=logger):
            pass

    def fit_with_data(self, datagen, logger=None):
        preds, targets, losses = zip(*self._process_data(datagen, train=True, logger=logger))
        preds, targets = torch.cat(preds, dim=0), torch.cat(targets, dim=0)
        return preds, targets, losses

    def fit_with_losses(self, datagen, logger=None):
        losses = [loss for _, _, loss in self._process_data(datagen, train=True, logger=logger)]
        return losses

    def predict_with_data(self, datagen, logger=None):
        with torch.no_grad():
            preds, targets, losses = zip(*self._process_data(datagen, train=False, logger=logger))
            preds, targets = torch.cat(preds, dim=0), torch.cat(targets, dim=0)
        return preds, targets, losses

    def predict_with_losses(self, datagen, logger=None):
        with torch.no_grad():
            losses = [loss for _, _, loss in self._process_data(datagen, train=False, logger=logger)]
        return losses



class DataParallelModel(TrainableModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parallel_apply = nn.DataParallel(*args, **kwargs)

    def forward(self, x):
        return self.parallel_apply(x)

    def loss(self, x, preds):
        return self.parallel_apply.module.loss(x, preds)


if __name__ == "__main__":
    IPython.embed()









