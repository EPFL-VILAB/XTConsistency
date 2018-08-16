
import os, sys, math, random, itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms

from utils import *
from models import TrainableModel
from logger import Logger, VisdomLogger
import IPython


class Network(TrainableModel):

    def __init__(self):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 10)
        self.to(DEVICE)

    def forward(self, x):

        x = F.relu(self.fc1(x.view(x.shape[0], -1)))
        x = F.dropout(x, 0.3, self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss(self, pred, target):
        return F.nll_loss(pred, target)



if __name__ == "__main__":

    model = Network()
    model.compile(torch.optim.SGD, lr=0.01, momentum=0.5, nesterov=True)
    logger = VisdomLogger("train", server='35.230.67.129', port=7000, env=JOB)
    logger.add_hook(lambda data: logger.step(), feature='loss', freq=500)

    def jointplot(data):
        data = np.stack([logger.data["train_loss"], logger.data["val_loss"]], axis=1)
        logger.plot(data, "loss", opts={'legend': ['train', 'val']})

    logger.add_hook(jointplot, feature='val_loss', freq=1)
    logger.add_hook(lambda data: logger.plot(data, "train_acc"), feature='accuracy', freq=2)
    logger.add_hook(lambda data: logger.plot(data, "test_acc"), feature='test_accuracy', freq=2)
    logger.add_hook(lambda x: 
        [print ("Saving model to /result/model.pth"),
        model.save("result/model.pth")],
        feature='loss', freq=400,
    )

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), 
            batch_size=32, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ])), 
            batch_size=32, shuffle=True)

    for epochs in range(0, 10):

        test_set = itertools.islice(val_loader, 1)
        test_images = torch.cat([x for x, y in test_set], dim=0)
        logger.images(test_images, "predictions")

        logger.update('epoch', epochs)

        preds, target, losses = model.fit_with_data(train_loader, logger=logger)
        accuracy = torch.sum(preds.argmax(dim=1) == target).float()/len(target)
        logger.update('train_loss', np.mean(losses))
        logger.update('accuracy', accuracy)

        preds, target, losses = model.predict_with_data(val_loader, logger=logger)
        accuracy = torch.sum(preds.argmax(dim=1) == target).float()/len(target)
        logger.update('val_loss', np.mean(losses))
        logger.update('test_accuracy', accuracy)

        logger.step()