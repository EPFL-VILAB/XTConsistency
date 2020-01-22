
import numpy as np
import random, sys, os, time, glob, math, itertools, json, copy
from collections import defaultdict, namedtuple
from functools import partial

import PIL
from PIL import Image
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.optim as optim
from torchvision import transforms

from utils import *
from models import DataParallelModel
from modules.unet import UNet, UNetOld2, UNetOld
from modules.percep_nets import Dense1by1Net
from modules.depth_nets import UNetDepth

import IPython

from PIL import ImageFilter
from skimage.filters import gaussian


class GaussianBulr(object):
    def __init__(self, radius):
        self.radius = radius
        self.filter = ImageFilter.GaussianBlur(radius)

    def __call__(self, im):
        return im.filter(self.filter)

    def __repr__(self):
        return 'GaussianBulr Filter with Radius {:d}'.format(self.radius)
'''
class GaussianBulr(object):
    def __init__(self, radius):
        self.radius = radius
        self.truncate = 1

    def __call__(self, im):
        return Image.fromarray((255.0*gaussian(np.array(im), sigma=self.radius, multichannel=True, truncate=self.truncate)).astype(np.uint8))

    def __repr__(self):
        return 'GaussianBulr Filter with Radius {:d}'.format(self.radius)
'''

""" Model definitions for launching new transfer jobs between tasks. """
model_types = {
    ('normal', 'principal_curvature'): lambda : Dense1by1Net(),
    ('normal', 'depth_zbuffer'): lambda : UNetDepth(),
    ('normal', 'reshading'): lambda : UNet(downsample=5),
    ('depth_zbuffer', 'normal'): lambda : UNet(downsample=6, in_channels=1, out_channels=3),
    ('reshading', 'normal'): lambda : UNet(downsample=4, in_channels=3, out_channels=3),
    ('sobel_edges', 'principal_curvature'): lambda : UNet(downsample=5, in_channels=1, out_channels=3),
    ('depth_zbuffer', 'principal_curvature'): lambda : UNet(downsample=4, in_channels=1, out_channels=3),
    ('principal_curvature', 'depth_zbuffer'): lambda : UNet(downsample=6, in_channels=3, out_channels=1),
    ('rgb', 'normal'): lambda : UNet(downsample=6),
    ('rgb', 'keypoints2d'): lambda : UNet(downsample=3, out_channels=1),
}

def get_model(src_task, dest_task):

    if isinstance(src_task, str) and isinstance(dest_task, str):
        src_task, dest_task = get_task(src_task), get_task(dest_task)

    if (src_task.name, dest_task.name) in model_types:
        return model_types[(src_task.name, dest_task.name)]()

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ImageTask):
        return UNet(downsample=5, in_channels=src_task.shape[0], out_channels=dest_task.shape[0])

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, ClassTask):
        return ResNet(in_channels=src_task.shape[0], out_channels=dest_task.classes)

    elif isinstance(src_task, ImageTask) and isinstance(dest_task, PointInfoTask):
        return ResNet(out_channels=dest_task.out_channels)

    return None



"""
Abstract task type definitions.
Includes Task, ImageTask, ClassTask, PointInfoTask, and SegmentationTask.
"""

class Task(object):
    """ General task output space"""
    def __init__(self, name,
            file_name=None, file_name_alt=None, file_ext="png", file_loader=None,
            plot_func=None
        ):

        super().__init__()
        self.name = name
        self.file_name, self.file_ext = file_name or name, file_ext
        self.file_name_alt = file_name_alt or self.file_name
        self.file_loader = file_loader or self.file_loader
        self.plot_func = plot_func or self.plot_func
        self.kind = name

    def norm(self, pred, target, batch_mean=True):
        if batch_mean:
            loss = ((pred - target)**2).mean()
        else:
            loss = ((pred - target)**2).mean(dim=1).mean(dim=1).mean(dim=1)
        return loss, (loss.mean().detach(),)

    def __call__(self, size=256):
        task = copy.deepcopy(self)
        return task

    def plot_func(self, data, name, logger, **kwargs):
        ### Non-image tasks cannot be easily plotted, default to nothing
        pass

    def file_loader(self, path, resize=None, seed=0, T=0):
        raise NotImplementedError()

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)


"""
Abstract task type definitions.
Includes Task, ImageTask, ClassTask, PointInfoTask, and SegmentationTask.
"""

class RealityTask(Task):
    """ General task output space"""

    def __init__(self, name, dataset, tasks=None, use_dataset=True, shuffle=False, batch_size=64):

        super().__init__(name=name)
        self.tasks = tasks if tasks is not None else \
            (dataset.dataset.tasks if hasattr(dataset, "dataset") else dataset.tasks)
        self.shape = (1,)
        if not use_dataset: return
        self.dataset, self.shuffle, self.batch_size = dataset, shuffle, batch_size
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=24, shuffle=self.shuffle, pin_memory=True
        )
        self.generator = cycle(loader)
        self.step()
        self.static = False

    @classmethod
    def from_dataloader(cls, name, loader, tasks):
        reality = cls(name, None, tasks, use_dataset=False)
        reality.loader = loader
        reality.generator = cycle(loader)
        reality.static = False
        reality.step()
        return reality

    @classmethod
    def from_static(cls, name, data, tasks):
        reality = cls(name, None, tasks, use_dataset=False)
        reality.task_data = {task: x.requires_grad_() for task, x in zip(tasks, data)}
        reality.static = True
        return reality

    def norm(self, pred, target, batch_mean=True):
        loss = torch.tensor(0.0, device=pred.device)
        return loss, (loss.detach(),)

    def step(self):
        self.task_data = {task: x.requires_grad_() for task, x in zip(self.tasks, next(self.generator))}

    def reload(self):
        loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=self.batch_size,
            num_workers=24, shuffle=self.shuffle, pin_memory=True
        )
        self.generator = cycle(loader)

class ImageTask(Task):
    """ Output space for image-style tasks """

    def __init__(self, *args, **kwargs):

        self.shape = kwargs.pop("shape", (3, 256, 256))
        self.mask_val = kwargs.pop("mask_val", -1.0)
        self.transform = kwargs.pop("transform", lambda x: x)
        self.resize = kwargs.pop("resize", self.shape[1])
        self.blur_radius = None
        self.image_transform = self.load_image_transform()
        super().__init__(*args, **kwargs)

    @staticmethod
    def build_mask(target, val=0.0, tol=1e-3):
        if target.shape[1] == 1:
            mask = ((target >= val - tol) & (target <= val + tol))
            mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
            return (~mask).expand_as(target)

        mask1 = (target[:, 0, :, :] >= val - tol) & (target[:, 0, :, :] <= val + tol)
        mask2 = (target[:, 1, :, :] >= val - tol) & (target[:, 1, :, :] <= val + tol)
        mask3 = (target[:, 2, :, :] >= val - tol) & (target[:, 2, :, :] <= val + tol)
        mask = (mask1 & mask2 & mask3).unsqueeze(1)
        mask = F.conv2d(mask.float(), torch.ones(1, 1, 5, 5, device=mask.device), padding=2) != 0
        return (~mask).expand_as(target)

    def norm(self, pred, target, batch_mean=True):
        mask = ImageTask.build_mask(target, val=self.mask_val)
        return super().norm(pred*mask.float(), target*mask.float(), batch_mean=batch_mean)

    def __call__(self, size=256, blur_radius=None):
        task = copy.deepcopy(self)
        task.shape = (3, size, size)
        task.resize = size
        task.blur_radius = blur_radius
        task.name +=  "blur" if blur_radius else str(size)
        task.base = self
        return task

    def plot_func(self, data, name, logger, resize=None, nrow=2):
        logger.images(data.clamp(min=0, max=1), name, nrow=nrow, resize=resize or self.resize)

    def file_loader(self, path, resize=None, crop=None, seed=0):
        image_transform = self.load_image_transform(resize=resize, crop=crop, seed=seed)
        return image_transform(Image.open(open(path, 'rb')))[0:3]

    def load_image_transform(self, resize=None, crop=None, seed=0):

        size = resize or self.resize
        random.seed(seed)
        crop_transform = lambda x: x
        if crop is not None:
            i = random.randint(0, size - crop)
            j = random.randint(0, size - crop)
            crop_transform = TF.crop(x, i, j, crop, crop)

        blur = [GaussianBulr(self.blur_radius)] if self.blur_radius else []
        return transforms.Compose(blur+[
            transforms.Resize(size, interpolation=PIL.Image.NEAREST),
            transforms.CenterCrop(size),
            crop_transform,
            transforms.ToTensor(),
            self.transform]
        )

class ImageClassTask(ImageTask):
    """ Output space for image-class segmentation tasks """

    def __init__(self, *args, **kwargs):

        self.classes = kwargs.pop("classes", (3, 256, 256))
        super().__init__(*args, **kwargs)

    def norm(self, pred, target):
        loss = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(target, dim=1))
        return loss, (loss.detach(),)

    def plot_func(self, data, name, logger, resize=None):
        _, idx = torch.max(data, dim=1)
        idx = idx.float()/16.0
        idx = idx.unsqueeze(1).expand(-1, 3, -1, -1)
        logger.images(idx.clamp(min=0, max=1), name, nrow=2, resize=resize or self.resize)

    def file_loader(self, path, resize=None):

        data = (self.image_transform(Image.open(open(path, 'rb')))*255.0).long()
        one_hot = torch.zeros((self.classes, data.shape[1], data.shape[2]))
        one_hot = one_hot.scatter_(0, data, 1)
        return one_hot


class ClassTask(Task):
    """ Output space for classification-style tasks """

    def __init__(self, *args, **kwargs):

        self.classes = kwargs.pop("classes")
        self.classes_file = kwargs.pop("classes_file")
        self.class_lookup = open(self.classes_file).readlines()

        kwargs["file_ext"] = kwargs.get("file_ext", "npy")
        super().__init__(*args, **kwargs)

    def norm(self, pred, target):
        # Input and target are BOTH logits
        loss = F.kl_div(pred, torch.exp(target))
        return loss, (loss.detach(),)

    def plot_func(self, data, name, logger):

        probs, idxs = torch.topk(data, 5, dim=1)
        probs = torch.exp(probs)
        probs, idxs = probs.cpu().data.numpy(), idxs.cpu().data.numpy()

        output = ""
        for i in range(0, probs.shape[0]):
            output += f"Example {i}: <br>"
            for j in range(0, probs.shape[1]):
                output += f"{self.class_lookup[idxs[i, j]]} ({probs[i, j]:0.5f})<br>"
            output += "<br>"

        logger.window(name, logger.visdom.text, output)

    def file_loader(self, path, resize=None):
        return torch.log(torch.tensor(np.load(path)).float())



class PointInfoTask(Task):
    """ Output space for point-info prediction tasks (what models do we evem use?) """

    def __init__(self, *args, **kwargs):

        self.point_type = kwargs.pop("point_type", "vanishing_points_gaussian_sphere")
        self.out_channels = 9
        super().__init__(*args, **kwargs)

    def plot_func(self, data, name, logger):
        logger.window(name, logger.visdom.text, str(data.data.cpu().numpy()))

    def file_loader(self, path, resize=None):
        points = json.load(open(path))[self.point_type]
        return np.array(points['x'] + points['y'] + points['z'])




"""
Current list of task definitions.
Accessible via tasks.{TASK_NAME} or get_task("{TASK_NAME}")
"""

def clamp_maximum_transform(x, max_val=8000.0):
    x = x.unsqueeze(0).float() / max_val
    return x[0].clamp(min=0, max=1)

def crop_transform(x, max_val=8000.0):
    x = x.unsqueeze(0).float() / max_val
    return x[0].clamp(min=0, max=1)

def sobel_transform(x):
    image = x.data.cpu().numpy().mean(axis=0)
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    sx = ndimage.sobel(blur, axis=0, mode='constant')
    sy = ndimage.sobel(blur, axis=1, mode='constant')
    sob = np.hypot(sx, sy)
    edge = torch.FloatTensor(sob).unsqueeze(0)
    return edge

def blur_transform(x, max_val=4000.0):
    if x.shape[0] == 1:
        x = x.squeeze(0)
    image = x.data.cpu().numpy()
    blur = ndimage.filters.gaussian_filter(image, sigma=2, )
    norm = torch.FloatTensor(blur).unsqueeze(0)**0.8 / (max_val**0.8)
    norm = norm.clamp(min=0, max=1)
    if norm.shape[0] != 1:
        norm = norm.unsqueeze(0)
    return norm

def sintel_depth_transform(x):
    x = (1.0/ (x+1/255.0))
    x = x / 255.0
    return x.clamp(min=0.0, max=1.0)

def get_task(task_name):
    return task_map[task_name]

def smooth_resolution_file_loader(path, resize=None, crop=None, seed=0, T=0):
    size = [128, 192, 256, 320, 384, 448, 512][T]
    image_transform = transforms.Compose([
        transforms.Resize(size, interpolation=PIL.Image.NEAREST),
        transforms.ToTensor()
    ])
    return image_transform(Image.open(open(path, 'rb')))[0:3]

def ds_us_file_loader(path, resize=None, crop=None, seed=0, T=0):
    image_transform = transforms.Compose([
        transforms.Resize(256, interpolation=PIL.Image.NEAREST),
        transforms.Resize(512, interpolation=PIL.Image.NEAREST),
        transforms.ToTensor()
    ])
    return image_transform(Image.open(open(path, 'rb')))[0:3]


tasks = [
    ImageTask('rgb'),
    ImageTask('normal', mask_val=0.502),
    ImageTask('principal_curvature', mask_val=0.0),
    ImageTask('depth_zbuffer',
        shape=(1, 256, 256),
        mask_val=1.0,
        transform=partial(clamp_maximum_transform, max_val=8000.0),
    ),
    ImageClassTask('segment_semantic',
        file_name_alt="segmentsemantic",
        shape=(16, 256, 256), classes=16,
    ),
    ImageTask('reshading', mask_val=0.0507),
    ImageTask('edge_occlusion',
        shape=(1, 256, 256),
        transform=partial(blur_transform, max_val=4000.0),
    ),
    ImageTask('sobel_edges',
        shape=(1, 256, 256),
        file_name='rgb',
        transform=sobel_transform,
    ),
    ImageTask('keypoints3d',
        shape=(1, 256, 256),
        transform=partial(clamp_maximum_transform, max_val=64131.0),
    ),
    ImageTask('keypoints2d',
        shape=(1, 256, 256),
        transform=partial(blur_transform, max_val=2000.0),
    ),
    # ClassTask('class_scene',
    #     file_name_alt="class_places",
    #     classes=365, classes_file="data/scene_classes.txt"
    # ),
    # ClassTask('class_object',
    #     classes=1000, classes_file="data/object_classes.txt"
    # ),
    # PointInfoTask('point_info'),
]


task_map = {task.name: task for task in tasks}
tasks = namedtuple('TaskMap', task_map.keys())(**task_map)

tasks.depth_zbuffer.sintel_depth = ImageTask('depth_sintel',
    shape=(1, 256, 256),
    mask_val=1.0,
    transform=sintel_depth_transform,
)


if __name__ == "__main__":
    IPython.embed()
