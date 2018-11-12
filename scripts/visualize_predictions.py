

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from fire import Fire

from utils import *
from plotting import *
from transfers import *
from models import TrainableModel, DataParallelModel
from logger import Logger, VisdomLogger
from datasets import ImageTaskDataset

from modules.unet import UNet
from losses import get_standardization_mixed_loss_fn

from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint
from functools import partial


def main(model_file="mount/shared/results_alpha_debugdepth_baseline1_5/model.pth"):
    curvature_weight = 0.0
    depth_weight = 0.0

    ### MODEL ###
    model = DataParallelModel.load(UNet().cuda(), model_file)
    
    model.compile(torch.optim.Adam, lr=3e-4, weight_decay=2e-6, amsgrad=True)
    print(model.forward(torch.randn(8, 3, 256, 256)).shape)

    ### LOGGING ###
    logger = VisdomLogger("train", env=JOB)

    ### DATA LOADING ###
    train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step = \
        load_data("rgb", "normal", batch_size=48)
    logger.images(test_images, "images", resize=128)
    logger.images(torch.cat(ood_images, dim=0), "ood_images", resize=128)
    plot_images(model, logger, test_set, ood_images, mask_val=0.502,
                    loss_models={"curvature": curvature_model, "depth": depth_model})



if __name__ == "__main__":
    Fire(main)
