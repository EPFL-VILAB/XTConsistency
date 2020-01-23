
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import random, sys, os, json, math

import torch
from torchvision import datasets, transforms, utils
import visdom

from utils import *
from utils import elapsed
import IPython

class BaseLogger(object):
    """ Logger class, with hooks for data features and plotting functions. """
    def __init__(self, name, verbose=True):

        self.name = name
        self.data = {}
        self.running_data = {}
        self.reset_running = {}
        self.verbose = verbose
        self.hooks = []

    def add_hook(self, hook, feature='epoch', freq=40):
        self.hooks.append((hook, feature, freq))

    def update(self, feature, x):
        x = torch.tensor(x).data.cpu().numpy().mean()

        self.data[feature] = self.data.get(feature, [])
        self.data[feature].append(x)
        if feature not in self.running_data or self.reset_running.pop(feature, False):
            self.running_data[feature] = []
        self.running_data[feature].append(x)

        for hook, hook_feature, freq in self.hooks:
            if feature == hook_feature and len(self.data[feature]) % freq == 0:
                hook(self, self.data)

    def step(self):
        buf = ""
        buf += f"({self.name}) "
        for feature in self.running_data.keys():
            if len(self.running_data[feature]) == 0: continue
            val = np.mean(self.running_data[feature])
            if float(val).is_integer():
                buf += f"{feature}: {int(val)}, "
            else:
                buf += f"{feature}: {val:0.4f}" + ", "
            self.reset_running[feature] = True
        buf += f" ... {elapsed():0.2f} sec"
        self.text (buf)

    def text(self, text, end="\n"):
        raise NotImplementedError()

    def plot(self, data, plot_name, opts={}):
        raise NotImplementedError()

    def images(self, data, image_name):
        raise NotImplementedError()

    def plot_feature(self, feature, opts={}):
        self.plot(self.data[feature], feature, opts)

    def plot_features(self, features, name, opts={}):
        stacked = np.stack([self.data[feature] for feature in features], axis=1)
        self.plot(stacked, name, opts={"legend": features})


class Logger(BaseLogger):

    def __init__(self, *args, **kwargs):
        self.results = kwargs.pop('results', 'output')
        super().__init__(*args, **kwargs)

    def text(self, text, end='\n'):
        print (text, end=end, flush=True)

    def plot(self, data, plot_name, opts={}):
        np.savez_compressed(f"{self.results}/{plot_name}.npz", data)
        plt.plot(data)
        plt.savefig(f"{self.results}/{plot_name}.jpg");
        plt.clf()


class VisdomLogger(BaseLogger):

    def __init__(self, *args, **kwargs):
        self.port = kwargs.pop('port', 7000)
        self.server = kwargs.pop('server', '35.229.22.191')
        # self.server = kwargs.pop('server', 'localhost')
        self.env = kwargs.pop('env', 'CH')
        print (f"Logging to environment {self.env}")
        self.visdom = visdom.Visdom(server="http://" + self.server, port=self.port, env=self.env)
        self.visdom.delete_env(self.env)
        self.windows = {}
        super().__init__(*args, **kwargs)
        self.save()
        self.add_hook(lambda logger, data: self.save(), feature="epoch", freq=1)

    def text(self, text, end='\n'):
        print (text, end=end)
        window, old_text = self.windows.get('text', (None, ""))
        if end == '\n': end = '<br>'
        display = old_text + text + end

        if window is not None:
            window = self.visdom.text (display, win=window, append=False)
        else:
            window = self.visdom.text (display)

        self.windows["text"] = window, display

    def window(self, plot_name, plot_func, *args, **kwargs):

        options = {'title': plot_name}
        options.update(kwargs.pop("opts", {}))
        window = self.windows.get(plot_name, None)
        if window is not None and self.visdom.win_exists(window):
            window = plot_func(*args, **kwargs, opts=options, win=window)
        else:
            window = plot_func(*args, **kwargs, opts=options)

        self.windows[plot_name] = window

    def plot(self, data, plot_name, opts={}):
        self.window(plot_name, self.visdom.line,
            np.array(data), X=np.array(range(len(data))), opts=opts
        )

    def histogram(self, data, plot_name, opts={}):
        self.window(plot_name, self.visdom.histogram, np.array(data), opts=opts)

    def scatter(self, X, Y, plot_name, opts={}):
        self.window(plot_name, self.visdom.scatter, np.stack([X, Y], axis=1), opts=opts)

    def bar(self, data, plot_name, opts={}):
        self.window(plot_name, self.visdom.bar, np.array(data), opts=opts)

    def save(self):
        self.visdom.save([self.env])

    def images(self, data, plot_name, opts={}, nrow=2, normalize=False, resize=64):

        transform = transforms.Compose([
                                    transforms.ToPILImage(),
                                    transforms.Resize(resize),
                                    transforms.ToTensor()])
        data = torch.stack([transform(x.cpu()) for x in data])
        data = utils.make_grid(data, nrow=nrow, normalize=normalize, pad_value=0)
        self.window(plot_name, self.visdom.image, np.array(data), opts=opts)

    def images_grouped(self, image_groups, plot_name, **kwargs):
        interleave = [y for x in zip(*image_groups) for y in x]
        self.images(interleave, plot_name, nrow=len(image_groups), **kwargs)


