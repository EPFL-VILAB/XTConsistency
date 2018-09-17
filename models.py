
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
    def fit_on_batch(self, data, target, loss_fn=None, metrics=[], train=True):
        
        loss_fn = loss_fn or self.loss

        self.train()
        self.zero_grad()
        self.optimizer.zero_grad()

        pred = self.forward(data)
        loss = loss_fn(pred, target.to(pred.device))

        with torch.no_grad():
            metric_data = tuple(metric(pred, target.to(pred.device)).detach() \
                        for metric in metrics)

        if train:
            loss.backward()
            self.optimizer.step()

        return pred, float(loss), metric_data

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
    def _process_data(self, datagen, loss_fn=None, metrics=[], train=True, logger=None):

        self.train(train)
        out = []
        for batch, y in datagen:
            y_pred, loss, metric_data = self.fit_on_batch(batch, y, loss_fn=loss_fn, metrics=metrics, train=train)
            if logger is not None: logger.update('loss', loss)
            yield ((y_pred.detach(), y.detach(), loss, metric_data))

    def fit(self, datagen, loss_fn=None, metrics=[], logger=None):
        for x in self._process_data(datagen, loss_fn=loss_fn, metrics=metrics, train=train, logger=logger):
            pass

    def fit_with_data(self, datagen, loss_fn=None, metrics=[], logger=None):
        preds, targets, losses, metrics = zip(*self._process_data(datagen, \
                loss_fn=loss_fn, metrics=metrics, train=True, logger=logger))
        preds, targets = torch.cat(preds, dim=0), torch.cat(targets, dim=0)
        metrics = zip(*metrics)
        return preds, targets, losses, metrics

    def fit_with_metrics(self, datagen, loss_fn=None, metrics=[], logger=None):
        metrics = [metrics for _, _, _, metrics in self._process_data(datagen, \
                loss_fn=loss_fn, metrics=metrics, train=True, logger=logger)]
        return zip(*metrics)

    def predict_with_data(self, datagen, loss_fn=None, metrics=[], logger=None):
        with torch.no_grad():
            preds, targets, losses, metrics = zip(*self._process_data(datagen, \
                    loss_fn=loss_fn, metrics=metrics, train=False, logger=logger))
            preds, targets = torch.cat(preds, dim=0), torch.cat(targets, dim=0)
            metrics = zip(*metrics)
        return preds, targets, losses, metrics

    def predict_with_metrics(self, datagen, loss_fn=None, metrics=[], logger=None):
        with torch.no_grad():
            metrics = [metrics for _, _, _, metrics in self._process_data(datagen, \
                    loss_fn=loss_fn, metrics=metrics, train=False, logger=logger)]
        return zip(*metrics)



class DataParallelModel(TrainableModel):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.parallel_apply = nn.DataParallel(*args, **kwargs)

    def forward(self, x):
        return self.parallel_apply(x)

    def loss(self, x, preds):
        return self.parallel_apply.module.loss(x, preds)

    @classmethod
    def load(cls, model=TrainableModel(), weights_file=None):
        model = cls(model)
        if weights_file is not None:
            model.load_state_dict(torch.load(weights_file))
        return model


if __name__ == "__main__":
    IPython.embed()









