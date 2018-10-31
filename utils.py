# utils.py

import numpy as np
import random, sys, os, time, glob, math, itertools

from sklearn.model_selection import train_test_split

import IPython
import PIL

EXPERIMENT, RESUME_JOB, BASE_DIR = open("scripts/jobinfo.txt").read().strip().split(', ')
JOB = "_".join(EXPERIMENT.split("_")[0:-1])

MODELS_DIR = f"{BASE_DIR}/shared/models"
DATA_DIR = f"{BASE_DIR}/data/taskonomy3"
RESULTS_DIR = f"{BASE_DIR}/shared/results_{EXPERIMENT}"
os.system(f"sudo mkdir {RESULTS_DIR}")

if BASE_DIR == "/":
    DATA_DIR = "/data"
    RESULTS_DIR = "/result"

print("Models dir: ", MODELS_DIR)
print("Results dir: ", RESULTS_DIR)
print("Data dir: ", DATA_DIR)

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


def build_mask(target, val=0.0, tol=1e-3, dilate=None):
    if target.shape[1] == 1:
        mask = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
        mask = mask.unsqueeze(1)
        mean = mask.data.float().mean()
        if dilate is not None:
            mask = F.conv2d(mask.float(), torch.ones(1, 1, dilate, dilate, device=mask.device), padding=dilate//2) != 0
        mask = (~mask).expand_as(target)
        return mask

    mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
    mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
    mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
    mask = (mask1 & mask2 & mask3).unsqueeze(1)
    if dilate is not None :
        mask = F.conv2d(mask.float(), torch.ones(1, 1, dilate, dilate, device=mask.device), padding=dilate//2) != 0

    mask = (~mask).expand_as(target)
    return mask


def load_data(source_task, dest_task, source_transforms=None, dest_transforms=None, batch_size=32, 
                resize=256, mask_val=0.502, dilate=5):
    
    from datasets import ImageTaskDataset, ImageDataset
    from torchvision import transforms

    test_buildings = ["almena", "albertville"]
    buildings = [file.split("/")[-1][:-7] for file in glob.glob(f"{DATA_DIR}/*_normal")]
    train_buildings, val_buildings = train_test_split(buildings, test_size=0.1)

    # building_tags = np.genfromtxt(open("data/train_val_test_fullplus.csv"), delimiter=",", dtype=str, skip_header=True)

    # test_buildings = ["almena", "mifflintown"]
    # train_buildings = [building for building, train, test, val in building_tags \
    #                         if train == "1" and building not in test_buildings]
    # val_buildings = [building for building, train, test, val in building_tags if val == "1"]

    resize = transforms.Compose([transforms.ToPILImage(), transforms.Resize(resize, interpolation=PIL.Image.NEAREST), 
                                    transforms.ToTensor()])

    def dilated_kernel(x):
        mask = build_mask(x.unsqueeze(0), mask_val, dilate=dilate)
        mask = resize(~mask[0])
        mask = (mask == 0).float()
        x = resize(x)
        x = x*mask.float() + mask_val*(1-mask.float())
        return x

    if source_transforms != None:
        source_transforms = transforms.Compose([transforms.Resize(resize), transforms.ToTensor()])
    else:
        source_transforms = transforms.Compose([transforms.ToTensor(), source_transforms or (lambda x: x), dilated_kernel])

    dest_transforms = transforms.Compose([transforms.ToTensor(), dest_transforms or (lambda x: x), dilated_kernel])

    train_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=train_buildings, source_transforms=source_transforms, dest_transforms=dest_transforms,
                         source_task=source_task, dest_task=dest_task),
        batch_size=batch_size,
        num_workers=64,
        shuffle=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=val_buildings, source_transforms=source_transforms, dest_transforms=dest_transforms,
                         source_task=source_task, dest_task=dest_task),
        batch_size=batch_size,
        num_workers=64,
        shuffle=True,
        pin_memory=True
    )
    test_loader1 = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=["almena"], source_transforms=source_transforms, dest_transforms=dest_transforms,
                         source_task=source_task, dest_task=dest_task, debug=True),
        batch_size=6,
        shuffle=False,
        pin_memory=True,
    )
    test_loader2 = torch.utils.data.DataLoader(
        ImageTaskDataset(buildings=["albertville"], source_transforms=source_transforms, dest_transforms=dest_transforms,
                         source_task=source_task, dest_task=dest_task, debug=True),
        batch_size=6,
        shuffle=False,
        pin_memory=True,
    )
    ood_loader = torch.utils.data.DataLoader(
        ImageDataset(data_dir="data/ood_images"),
        batch_size=10,
        shuffle=False,
        pin_memory=True
    )
    train_step = int(2248616 // (100 * batch_size))
    val_step = int(245592 // (100 * batch_size))
    print("Train step: ", train_step)
    print("Val step: ", val_step)

    train_loader, val_loader = cycle(train_loader), cycle(val_loader)
    test_set = list(itertools.islice(test_loader1, 1)) + list(itertools.islice(test_loader2, 1))
    test_images = torch.cat([x for x, y in test_set], dim=0)
    ood_images = list(itertools.islice(ood_loader, 1))

    return train_loader, val_loader, test_set, test_images, ood_images, train_step, val_step


def plot_images(model, logger, test_set, ood_images=None, mask_val=0.502, loss_models={}):
    preds, targets, losses, _ = model.predict_with_data(test_set)
    test_masks = build_mask(targets, mask_val, tol=1e-3)
    logger.images(test_masks.float(), "masks", resize=256)
    logger.images(preds.clamp(min=0, max=1), "predictions", nrow=2, resize=256)
    logger.images(targets.clamp(min=0, max=1), "targets", nrow=2, resize=256)
    logger.images(targets.clamp(min=0, max=1)*test_masks.float() + (1-mask_val)*(1 - test_masks.float()), "targets_masked", nrow=2, resize=256)

    if ood_images is not None:
        ood_preds = model.predict(ood_images)
        logger.images(ood_preds, "ood_predictions", nrow=2, resize=256)

    for name, loss_model in loss_models.items():
        with torch.no_grad():
            curvature_preds = loss_model(preds)
            curvature_targets = loss_model(targets)
            logger.images(curvature_preds.clamp(min=0, max=1), f"{name}_predictions", resize=128)
            logger.images(curvature_targets.clamp(min=0, max=1), f"{name}_targets", resize=128)
