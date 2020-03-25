import os, sys, math, random, itertools
import numpy as np
import scipy
from collections import defaultdict
from tqdm import tqdm
import pickle as pkl
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph
#from logger import Logger, VisdomLogger
from datasets import TaskDataset, load_train_val, load_test, load_ood, ImageDataset
from task_configs import tasks, RealityTask

from modules.resnet import ResNet
from modules.unet import UNet, UNetOld
from functools import partial
from fire import Fire

import IPython
import pdb

MAX_BLUR_SIGMA = 10

def main(
    loss_config="conservative_full",
    mode="standard", visualize=False,
    pretrained=True, finetuned=False, fast=False, batch_size=None,
    ood_batch_size=None, subset_size=None,
    cont=None,
    cont_gan=None, pre_gan=None, max_epochs=800, use_baseline=False, use_l1=False, num_workers=32, data_dir=None, save_dir='mount/shared/', **kwargs,
):

    # CONFIG
    batch_size = batch_size or (4 if fast else 64)
    energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

    # Setting up OOD
    if data_dir is None or data_dir == 'TASKONOMY_TEST':
        buildings = ["almena", "albertville"]
        train_subset_dataset = TaskDataset(buildings, tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature])
    elif data_dir == 'APOLLOSCAPE':
        train_subset_dataset = ImageDataset(data_dir=f"{BASE_DIR}/apolloscape/ResizeImage/Record*/")
    elif data_dir == 'COCODOOM':
        train_subset_dataset = ImageDataset(data_dir=f"{BASE_DIR}/cocodoom/run*/map*/rgb")
    elif data_dir == 'PR_VIDEO':
        train_subset_dataset = ImageDataset(data_dir=f"{BASE_DIR}/shared/assets/input_frames", tasks=[tasks.rgb, tasks.filename])
    else:
        train_subset_dataset = ImageDataset(data_dir=data_dir)
        data_dir = 'CUSTOM'
    train_subset = RealityTask("train_subset", train_subset_dataset, batch_size=batch_size, shuffle=False)

    if subset_size is None:
        subset_size = len(train_subset_dataset)
    subset_size = min(subset_size, len(train_subset_dataset))

    # GRAPH
    realities = [train_subset] #train_subset [ood]
    edges = []
    for t in energy_loss.tasks:
        if t != tasks.rgb:
            edges.append((tasks.rgb, t))
            edges.append((tasks.rgb, tasks.normal))


    graph = TaskGraph(tasks=energy_loss.tasks + [train_subset],
                      finetuned=finetuned,
                      freeze_list=energy_loss.freeze_list, lazy=True,
                      initialize_from_transfer=False,
                      )
    print('tasks:', energy_loss.tasks + realities)
    print('file', cont)
    graph.load_weights(cont)
    graph.compile(optimizer=None)

    # Add consistency links
    if tasks.reshading in energy_loss.tasks:
        rgb2reshading_graph = TaskGraph(tasks=[tasks.rgb, tasks.reshading, tasks.normal], lazy=True,
                      initialize_from_transfer=False, edges=[(tasks.rgb, tasks.reshading), (tasks.reshading, tasks.normal)]
                      )
        rgb2reshading_graph.load_weights(f'{SHARED_DIR}/results_CH_lbp_all_reshadingtarget_gradnorm_unnormalizedmse_imagenet_nosqerror_nosqinitauglr_dataaug_1/graph.pth')
        rgb2reshading_graph.compile(optimizer=None)
        # print(rgb2reshading_graph.edge_map.keys())
        graph.edge_map[f"('{tasks.rgb}', '{tasks.reshading}')"] = rgb2reshading_graph.edge_map[f"('{tasks.rgb}', '{tasks.reshading}')"]
        del rgb2reshading_graph


    if tasks.depth_zbuffer in energy_loss.tasks:
        rgb2depth_graph = TaskGraph(tasks=[tasks.rgb, tasks.depth_zbuffer, tasks.normal], lazy=True,
                      initialize_from_transfer=False, edges=[(tasks.rgb, tasks.depth_zbuffer), (tasks.depth_zbuffer, tasks.normal)]
                      )
        rgb2depth_graph.load_weights(f'{SHARED_DIR}/results_CH_lbp_all_depthtarget_gradnorm_unnormalizedmse_imagenet_nosqerror_nosqinitauglr_dataaug_1/graph.pth')
        rgb2depth_graph.compile(optimizer=None)
        print(rgb2depth_graph.edge_map.keys())
        graph.edge_map[f"('{tasks.rgb}', '{tasks.depth_zbuffer}')"] = rgb2depth_graph.edge_map[f"('{tasks.rgb}', '{tasks.depth_zbuffer}')"]
        del rgb2depth_graph


    rgb2normal_graph = TaskGraph(tasks=[tasks.rgb, tasks.normal], lazy=True,
                      initialize_from_transfer=False, edges=[(tasks.rgb, tasks.normal)]
                      )

    rgb2normal_graph.load_weights(f'{SHARED_DIR}/results_CH_lbp_all_normaltarget_gradnorm_unnormalizedmse_imagenet_nosqerror_nosqinitauglr_dataaug_1/graph.pth')
    rgb2normal_graph.compile(optimizer=None)
    print(rgb2normal_graph.edge_map.keys())
    graph.edge_map[f"('{tasks.rgb}', '{tasks.normal}')"] = rgb2normal_graph.edge_map[f"('{tasks.rgb}', '{tasks.normal}')"]
    del rgb2normal_graph


    best_ood_val_loss = float('inf')
    energy_losses = []
    mse_losses = []
    pearsonr_vals = []
    percep_losses = defaultdict(list)
    pearson_percep = defaultdict(list)

    energy_mean_by_blur = []
    energy_std_by_blur = []
    error_mean_by_blur = []
    error_std_by_blur = []


    blur_size = 0
    train_subset.reload()
    tasks.rgb.jpeg_quality = blur_size


    energy_losses = []
    error_losses = []

    energy_losses_all = []
    energy_losses_headings = []

    fnames = []
    # Compute energies
    for epochs in tqdm(range(subset_size // batch_size)):
        with torch.no_grad():
            losses = energy_loss(graph, realities=[train_subset], reduce=False, use_l1=use_l1)

            if len(energy_losses_headings) == 0:
                energy_losses_headings = sorted([loss_name for loss_name in losses if 'percep' in loss_name])

            all_perceps = [losses[loss_name].cpu().numpy() for loss_name in energy_losses_headings]
            direct_losses = [losses[loss_name].cpu().numpy() for loss_name in losses if 'direct' in loss_name]

            if len(all_perceps) > 0:
                energy_losses_all += [all_perceps]
                all_perceps = np.stack(all_perceps)
                energy_losses += list(all_perceps.mean(0))

            if len(direct_losses) > 0:
                direct_losses = np.stack(direct_losses)
                error_losses += list(direct_losses.mean(0))

            if False:
                fnames += train_subset.task_data[tasks.filename]
        train_subset.step()


    # Log losses
    if len(energy_losses) > 0:
        energy_losses = np.array(energy_losses)
        print(f'energy = {energy_losses.mean()}')

        energy_mean_by_blur += [energy_losses.mean()]
        energy_std_by_blur += [np.std(energy_losses)]

    if len(error_losses) > 0:
        error_losses = np.array(error_losses)
        print(f'error = {error_losses.mean()}')

        error_mean_by_blur += [error_losses.mean()]
        error_std_by_blur += [np.std(error_losses)]

    # save to csv
    save_error_losses = error_losses if len(error_losses) > 0 else [0] * subset_size
    save_energy_losses = energy_losses if len(energy_losses) > 0 else [0] * subset_size

    percep_losses = { k: v for k, v in zip(energy_losses_headings, np.concatenate(energy_losses_all, axis=-1))}
    df = pd.DataFrame(both(
                    {'energy': save_energy_losses, 'error': save_error_losses },
                    percep_losses
    ))
    df.to_csv(f"{save_dir}/{data_dir}.csv", mode='w', header=True)




if __name__ == "__main__":
    Fire(main)
