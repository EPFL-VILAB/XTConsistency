import os, sys, math, random, itertools
from functools import partial
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

from utils import *
from plotting import *
from task_configs import tasks, get_task
from transfers import functional_transfers, finetuned_transfers, get_transfer_name, Transfer
from datasets import TaskDataset, load_train_val

import IPython

import pdb

def get_energy_loss(
    config="", mode="standard",
    pretrained=True, finetuned=True, **kwargs,
):
    """ Loads energy loss from config dict. """
    if isinstance(mode, str):
        mode = {
            "standard": EnergyLoss,
            "curriculum": CurriculumEnergyLoss,
            "curriclat": CurricLATEnergyLoss,
            "normalized": NormalizedEnergyLoss,
            "percepnorm": PercepNormEnergyLoss,
            "lat": LATEnergyLoss,
            "winrate": WinRateEnergyLoss,
        }[mode]
    return mode(**energy_configs[config],
        pretrained=pretrained, finetuned=finetuned, **kwargs
    )


energy_configs = {

    "multiperceptual_depth": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.depth_zbuffer],
            "n(x)": [tasks.rgb, tasks.depth_zbuffer],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.reshading],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.normal],
            "reshading": [tasks.reshading],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.depth_zbuffer, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.principal_curvature],
            "s(y^)": [tasks.depth_zbuffer, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.sobel_edges],
            "g(y^)": [tasks.depth_zbuffer, tasks.normal],
            "g(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.normal],
            "nr(y^)": [tasks.depth_zbuffer, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.reshading],
            "Nk2(y^)": [tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints2d],
            "Nk3(y^)": [tasks.depth_zbuffer, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.keypoints3d],
            "nEO(y^)": [tasks.depth_zbuffer, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.depth_zbuffer, tasks.edge_occlusion],
        },
        "freeze_list": [
            [tasks.depth_zbuffer, tasks.principal_curvature],
            [tasks.depth_zbuffer, tasks.sobel_edges],
            [tasks.depth_zbuffer, tasks.normal],
            [tasks.depth_zbuffer, tasks.reshading],
            [tasks.depth_zbuffer, tasks.keypoints3d],
            [tasks.depth_zbuffer, tasks.keypoints2d],
            [tasks.depth_zbuffer, tasks.edge_occlusion],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_normal": {
                ("train", "val"): [
                    ("d(x)", "depth"),
                ],
            },
            "percep_reshading": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_reshading": {
                ("train", "val"): [
                    ("r(x)", "reshading"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "keypoints2d"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "multiperceptual_reshading": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.reshading],
            "n(x)": [tasks.rgb, tasks.reshading],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.normal],
            "r(x)": [tasks.rgb, tasks.depth_zbuffer],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.normal],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.reshading, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.reshading, tasks.principal_curvature],
            "s(y^)": [tasks.reshading, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.reshading, tasks.sobel_edges],
            "g(y^)": [tasks.reshading, tasks.normal],
            "g(n(x))": [tasks.rgb, tasks.reshading, tasks.normal],
            "nr(y^)": [tasks.reshading, tasks.depth_zbuffer],
            "nr(n(x))": [tasks.rgb, tasks.reshading, tasks.depth_zbuffer],
            "Nk2(y^)": [tasks.reshading, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.reshading, tasks.keypoints2d],
            "Nk3(y^)": [tasks.reshading, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.reshading, tasks.keypoints3d],
            "nEO(y^)": [tasks.reshading, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.reshading, tasks.edge_occlusion],
        },
        "freeze_list": [
            [tasks.reshading, tasks.principal_curvature],
            [tasks.reshading, tasks.sobel_edges],
            [tasks.reshading, tasks.normal],
            [tasks.reshading, tasks.depth_zbuffer],
            [tasks.reshading, tasks.keypoints3d],
            [tasks.reshading, tasks.keypoints2d],
            [tasks.reshading, tasks.edge_occlusion],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_normal": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_normal": {
                ("train", "val"): [
                    ("d(x)", "reshading"),
                ],
            },
            "percep_depth_zbuffer": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_depth_zbuffer": {
                ("train", "val"): [
                    ("r(x)", "depth"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "keypoints2d"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
        },
        "plots": {
            "": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                ]
            ),
        },
    },

    "multiperceptual": {
        "paths": {
            "x": [tasks.rgb],
            "y^": [tasks.normal],
            "n(x)": [tasks.rgb, tasks.normal],
            "RC(x)": [tasks.rgb, tasks.principal_curvature],
            "a(x)": [tasks.rgb, tasks.sobel_edges],
            "d(x)": [tasks.rgb, tasks.depth_zbuffer],
            "r(x)": [tasks.rgb, tasks.reshading],
            "k2(x)": [tasks.rgb, tasks.keypoints2d],
            "k3(x)": [tasks.rgb, tasks.keypoints3d],
            "EO(x)": [tasks.rgb, tasks.edge_occlusion],
            "curv": [tasks.principal_curvature],
            "edge": [tasks.sobel_edges],
            "depth": [tasks.depth_zbuffer],
            "reshading": [tasks.reshading],
            "keypoints2d": [tasks.keypoints2d],
            "keypoints3d": [tasks.keypoints3d],
            "edge_occlusion": [tasks.edge_occlusion],
            "f(y^)": [tasks.normal, tasks.principal_curvature],
            "f(n(x))": [tasks.rgb, tasks.normal, tasks.principal_curvature],
            "s(y^)": [tasks.normal, tasks.sobel_edges],
            "s(n(x))": [tasks.rgb, tasks.normal, tasks.sobel_edges],
            "g(y^)": [tasks.normal, tasks.depth_zbuffer],
            "g(n(x))": [tasks.rgb, tasks.normal, tasks.depth_zbuffer],
            "nr(y^)": [tasks.normal, tasks.reshading],
            "nr(n(x))": [tasks.rgb, tasks.normal, tasks.reshading],
            "Nk2(y^)": [tasks.normal, tasks.keypoints2d],
            "Nk2(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints2d],
            "Nk3(y^)": [tasks.normal, tasks.keypoints3d],
            "Nk3(n(x))": [tasks.rgb, tasks.normal, tasks.keypoints3d],
            "nEO(y^)": [tasks.normal, tasks.edge_occlusion],
            "nEO(n(x))": [tasks.rgb, tasks.normal, tasks.edge_occlusion],
        },
        "freeze_list": [
            [tasks.normal, tasks.principal_curvature],
            [tasks.normal, tasks.sobel_edges],
            [tasks.normal, tasks.depth_zbuffer],
            [tasks.normal, tasks.reshading],
            [tasks.normal, tasks.keypoints2d],
            [tasks.normal, tasks.keypoints3d],
            [tasks.normal, tasks.edge_occlusion],
        ],
        "losses": {
            "mse": {
                ("train", "val"): [
                    ("n(x)", "y^"),
                ],
            },
            "percep_curv": {
                ("train", "val"): [
                    ("f(n(x))", "f(y^)"),
                ],
            },
            "direct_curv": {
                ("train", "val"): [
                    ("RC(x)", "curv"),
                ],
            },
            "percep_edge": {
                ("train", "val"): [
                    ("s(n(x))", "s(y^)"),
                ],
            },
            "direct_edge": {
                ("train", "val"): [
                    ("a(x)", "s(y^)"),
                ],
            },
            "percep_depth": {
                ("train", "val"): [
                    ("g(n(x))", "g(y^)"),
                ],
            },
            "direct_depth": {
                ("train", "val"): [
                    ("d(x)", "depth"),
                ],
            },
            "percep_reshading": {
                ("train", "val"): [
                    ("nr(n(x))", "nr(y^)"),
                ],
            },
            "direct_reshading": {
                ("train", "val"): [
                    ("r(x)", "reshading"),
                ],
            },
            "percep_keypoints2d": {
                ("train", "val"): [
                    ("Nk2(n(x))", "Nk2(y^)"),
                ],
            },
            "direct_keypoints2d": {
                ("train", "val"): [
                    ("k2(x)", "Nk2(y^)"),
                ],
            },
            "percep_keypoints3d": {
                ("train", "val"): [
                    ("Nk3(n(x))", "Nk3(y^)"),
                ],
            },
            "direct_keypoints3d": {
                ("train", "val"): [
                    ("k3(x)", "keypoints3d"),
                ],
            },
            "percep_edge_occlusion": {
                ("train", "val"): [
                    ("nEO(n(x))", "nEO(y^)"),
                ],
            },
            "direct_edge_occlusion": {
                ("train", "val"): [
                    ("EO(x)", "edge_occlusion"),
                ],
            },
        },
        "plots": {
            "ID": dict(
                size=256,
                realities=("test", "ood"),
                paths=[
                    "x",
                    "y^",
                    "n(x)",
                    "f(y^)",
                    "f(n(x))",
                    "s(y^)",
                    "s(n(x))",
                    "g(y^)",
                    "g(n(x))",
                    "nr(n(x))",
                    "nr(y^)",
                    "Nk2(y^)",
                    "Nk2(n(x))",
                    "Nk3(y^)",
                    "Nk3(n(x))",
                    "nEO(y^)",
                    "nEO(n(x))",
                ]
            ),
        },
    },

}



def coeff_hook(coeff):
    def fun1(grad):
        return coeff*grad.clone()
    return fun1


class EnergyLoss(object):

    def __init__(self, paths, losses, plots,
        pretrained=True, finetuned=False, freeze_list=[]
    ):

        self.paths, self.losses, self.plots = paths, losses, plots
        self.freeze_list = [str((path[0].name, path[1].name)) for path in freeze_list]
        self.metrics = {}

        self.tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    # IPython.embed()
                    self.tasks += self.paths[path1] + self.paths[path2]

        for name, config in self.plots.items():
            for path in config["paths"]:
                self.tasks += self.paths[path]
        self.tasks = list(set(self.tasks))

    def compute_paths(self, graph, reality=None, paths=None):
        path_cache = {}
        paths = paths or self.paths
        path_values = {
            name: graph.sample_path(path,
                reality=reality, use_cache=True, cache=path_cache,
            ) for name, path in paths.items()
        }
        del path_cache
        return {k: v for k, v in path_values.items() if v is not None}

    def get_tasks(self, reality):
        tasks = []
        for _, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                if reality in realities:
                    for path1, path2 in losses:
                        tasks += [self.paths[path1][0], self.paths[path2][0]]

        for name, config in self.plots.items():
            if reality in config["realities"]:
                for path in config["paths"]:
                    tasks += [self.paths[path][0]]

        return list(set(tasks))

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None, batch_mean=True):
        loss = {}
        for reality in realities:
            loss_dict = {}
            losses = []
            all_loss_types = set()
            for loss_type, loss_item in self.losses.items():
                all_loss_types.add(loss_type)
                loss_dict[loss_type] = []
                for realities_l, data in loss_item.items():
                    if reality.name in realities_l:
                        loss_dict[loss_type] += data
                        if loss_types is not None and loss_type in loss_types:
                            losses += data

            path_values = self.compute_paths(graph,
                paths={
                    path: self.paths[path] for path in \
                    set(path for paths in losses for path in paths)
                    },
                reality=reality)

            if reality.name not in self.metrics:
                self.metrics[reality.name] = defaultdict(list)

            for loss_type, losses in sorted(loss_dict.items()):
                if loss_type not in (loss_types or all_loss_types):
                    continue
                if loss_type != 'gan':
                    if loss_type not in loss:
                        loss[loss_type] = 0
                    for path1, path2 in losses:

                        output_task = self.paths[path1][-1]
                        if "direct" in loss_type:
                            with torch.no_grad():
                                path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean)
                                loss[loss_type] += path_loss
                        else:
                            path_loss, _ = output_task.norm(path_values[path1], path_values[path2], batch_mean=batch_mean)
                            loss[loss_type] += path_loss
                            self.metrics[reality.name][loss_type +" : "+path1 + " -> " + path2] += [path_loss.mean().detach().cpu()]

                elif loss_type == 'gan':

                    for path1, path2 in losses:
                        if 'gan'+path1+path2 not in loss:
                            loss['disgan'+path1+path2] = 0
                            loss['graphgan'+path1+path2] = 0

                        ## Hack: detach() first pass so only OOD is updated
                        if path_values[path1] is None: continue
                        if path_values[path2] is None: continue
                        logit_path1 = discriminator(path_values[path1].detach())

                        ## Progressively increase GAN trade-off
                        #coeff = np.float(2.0 / (1.0 + np.exp(-10.0*self.train_iter / 10000.0)) - 1.0)
                        coeff = 0.1
                        path_value2 = path_values[path2] * 1.0
                        if reality.name == 'train':
                            path_value2.register_hook(coeff_hook(coeff))
                        logit_path2 = discriminator(path_value2)
                        binary_label = torch.Tensor([1]*logit_path1.size(0)+[0]*logit_path2.size(0)).float().cuda()
                        # print ("In BCE loss for gan: ", reality, logit_path1.mean(), logit_path2.mean())

                        gan_loss = nn.BCEWithLogitsLoss(size_average=True)(torch.cat((logit_path1,logit_path2), dim=0).view(-1), binary_label)
                        self.metrics[reality.name]['gan : ' + path1 + " -> " + path2] += [gan_loss.detach().cpu()]
                        loss['disgan'+path1+path2] -= gan_loss
                        #binary_label_ood = torch.Tensor([0.5]*(logit_path1.size(0)+logit_path2.size(0))).float().cuda()
                        #gan_loss_ood = nn.BCELoss(size_average=True)(nn.Sigmoid()(torch.cat((logit_path1,logit_path2), dim=0).view(-1)), binary_label_ood)
                        #loss['graphgan'+path1+path2] += gan_loss_ood
                else:
                    raise Exception('Loss {} not implemented.'.format(loss_type))

        return loss

    def logger_hooks(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    name = loss_type+" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            def jointplot(logger, data, name=name, realities=realities):

                names = [f"{reality}_{name}" for reality in realities]
                if not all(x in data for x in names):
                    return
                data = np.stack([data[x] for x in names], axis=1)
                logger.plot(data, name, opts={"legend": names})

            logger.add_hook(partial(jointplot, name=name, realities=realities), feature=f"{realities[-1]}_{name}", freq=1)


    def logger_update(self, logger):

        name_to_realities = defaultdict(list)
        for loss_type, loss_item in self.losses.items():
            for realities, losses in loss_item.items():
                for path1, path2 in losses:
                    name = loss_type +" : "+path1 + " -> " + path2
                    name_to_realities[name] += list(realities)

        for name, realities in name_to_realities.items():
            for reality in realities:
                # IPython.embed()
                if reality not in self.metrics: continue
                if name not in self.metrics[reality]: continue
                if len(self.metrics[reality][name]) == 0: continue

                logger.update(
                    f"{reality}_{name}",
                    torch.mean(torch.stack(self.metrics[reality][name])),
                )
        self.metrics = {}

    def plot_paths(self, graph, logger, realities=[], plot_names=None, prefix=""):

        realities_map = {reality.name: reality for reality in realities}
        for name, config in (plot_names or self.plots.items()):
            paths = config["paths"]
            realities = config["realities"]
            images = [[] for _ in range(0, len(paths))]
            for reality in realities:
                with torch.no_grad():
                    path_values = self.compute_paths(graph, paths={path: self.paths[path] for path in paths}, reality=realities_map[reality])
                shape = list(path_values[list(path_values.keys())[0]].shape)
                shape[1] = 3
                for i, path in enumerate(paths):
                    X = path_values.get(path, torch.zeros(shape, device=DEVICE))
                    images[i].append(X.clamp(min=0, max=1).expand(*shape))

            for i in range(0, len(paths)):
                images[i] = torch.cat(images[i], dim=0)

            logger.images_grouped(images,
                f"{prefix}_{name}_[{', '.join(realities)}]_[{', '.join(paths)}]",
                resize=config["size"],
            )

    def __repr__(self):
        return str(self.losses)






class NormalizedEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.loss_weights = {}
        self.update_freq = kwargs.pop("update_freq", 30)
        self.iter = 0
        super().__init__(*args, **kwargs)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        if self.iter % self.update_freq == 0:
            self.loss_weights = {key: 1.0/loss.detach() for key, loss in loss_dict.items()}
        self.iter += 1
        return {key: loss * self.loss_weights[key] for key, loss in loss_dict.items()}





class CurriculumEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.percep_weight = 0.0
        self.percep_step = kwargs.pop("percep_step", 0.1)
        super().__init__(*args, **kwargs)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        loss_dict["percep"] = loss_dict["percep"] * self.percep_weight
        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)
        self.percep_weight += self.percep_step
        logger.text (f'Current percep weight: {self.percep_weight}')






class PercepNormEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        # everything in loss_dict["percep"] should be normalized via a direct path
        # everything in loss_dict["direct"] should be used as a normalizer
        # everything in loss_dict["mse"] should be standardized to 1

        percep_losses = [key[7:] for key in loss_dict if key[0:7] == "percep_"]
        for key in percep_losses:
            # print (key, loss_dict[f"percep_{key}"], loss_dict[f"direct_{key}"])
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"] / loss_dict[f"direct_{key}"]
            # print (loss_dict[f"percep_{key}"])
            loss_dict.pop(f"direct_{key}")

        loss_dict["mse"] = loss_dict["mse"] / loss_dict["mse"].detach()
        return loss_dict







class LATEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print (self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):

        loss_types = ["mse"] + [("percep_" + loss) for loss in self.chosen_losses] + [("direct_" + loss) for loss in self.chosen_losses]
        print (self.chosen_losses)
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        # everything in loss_dict["percep"] should be normalized via a direct path
        # everything in loss_dict["direct"] should be used as a normalizer
        # everything in loss_dict["mse"] should be standardized to 1

        for key in self.chosen_losses:
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"] / loss_dict[f"direct_{key}"].detach()
            loss_dict.pop(f"direct_{key}")
            self.running_stats[key] = loss_dict[f"percep_{key}"].detach().cpu().item()

        loss_dict["mse"] = loss_dict["mse"] / loss_dict["mse"].detach()
        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)
        if self.random_select or len(self.running_stats) < len(self.percep_losses):
            self.chosen_losses = random.sample(self.percep_losses, self.k)
        else:
            self.chosen_losses = sorted(self.running_stats, key=self.running_stats.get, reverse=True)[:self.k]

        logger.text (f"Chosen losses: {self.chosen_losses}")




class CurricLATEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.percep_weight = 0.0
        self.percep_step = kwargs.pop("percep_step", 0.1)
        self.running_stats = {}

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print (self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):

        loss_types = ["mse"] + [("percep_" + loss) for loss in self.chosen_losses] + [("direct_" + loss) for loss in self.chosen_losses]
        print (self.chosen_losses)
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types)
        # everything in loss_dict["percep"] should be normalized via a direct path
        # everything in loss_dict["direct"] should be used as a normalizer
        # everything in loss_dict["mse"] should be standardized to 1

        for key in self.chosen_losses:
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"] / loss_dict[f"direct_{key}"].detach()
            loss_dict.pop(f"direct_{key}")
            self.running_stats[key] = loss_dict[f"percep_{key}"].detach().cpu().item()
            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"] * (0.0035 * self.percep_weight)

        loss_dict["mse"] = loss_dict["mse"]
        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)
        if self.random_select or len(self.running_stats) < len(self.percep_losses):
            self.chosen_losses = random.sample(self.percep_losses, self.k)
        else:
            self.chosen_losses = sorted(self.running_stats, key=self.running_stats.get, reverse=True)[:self.k]

        logger.text (f"Chosen losses: {self.chosen_losses}")
        self.percep_weight += self.percep_step
        logger.text (f"Percep weight: {self.percep_weight}")






class WinRateEnergyLoss(EnergyLoss):

    def __init__(self, *args, **kwargs):
        self.k = kwargs.pop('k', 3)
        self.random_select = kwargs.pop('random_select', False)
        self.running_stats = {}

        super().__init__(*args, **kwargs)

        self.percep_losses = [key[7:] for key in self.losses.keys() if key[0:7] == "percep_"]
        print (self.percep_losses)
        self.chosen_losses = random.sample(self.percep_losses, self.k)

    def __call__(self, graph, discriminator=None, realities=[], loss_types=None):

        loss_types = ["mse"] + [("percep_" + loss) for loss in self.chosen_losses] + [("direct_" + loss) for loss in self.chosen_losses]
        print (self.chosen_losses)
        loss_dict = super().__call__(graph, discriminator=discriminator, realities=realities, loss_types=loss_types, batch_mean=False)
        # everything in loss_dict["percep"] should be normalized via a direct path
        # everything in loss_dict["direct"] should be used as a normalizer
        # everything in loss_dict["mse"] should be standardized to 1

        for key in self.chosen_losses:
            winrate = torch.mean((loss_dict[f"percep_{key}"] > loss_dict[f"direct_{key}"]).float())
            winrate = winrate.detach().cpu().item()

            if winrate < 1.0:
                self.running_stats[key] = winrate

            loss_dict[f"percep_{key}"] = loss_dict[f"percep_{key}"].mean() / loss_dict[f"direct_{key}"].mean().detach()
            loss_dict.pop(f"direct_{key}")

        print (self.running_stats)
        loss_dict["mse"] = loss_dict["mse"].mean() / loss_dict["mse"].mean().detach()
        return loss_dict

    def logger_update(self, logger):
        super().logger_update(logger)
        # TODO: randomly select losses if they have the same winrate?
        if self.random_select or len(self.running_stats) < len(self.percep_losses):
            self.chosen_losses = random.sample(self.percep_losses, self.k)
        else:
            self.chosen_losses = sorted(self.running_stats, key=self.running_stats.get, reverse=True)[:self.k]

        logger.text (f"Chosen losses: {self.chosen_losses}")


