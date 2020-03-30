import os, sys, math, random, itertools
import numpy as np
import scipy
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from plotting import *
from energy import get_energy_loss
from graph import TaskGraph
from datasets import TaskDataset, load_train_val, load_test, load_ood, ImageDataset
from task_configs import tasks, RealityTask

from functools import partial
from fire import Fire

import IPython
import pdb


def main(
    loss_config="conservative_full",
    mode="standard",
    pretrained=True, finetuned=False, batch_size=16,
    ood_batch_size=None, subset_size=None,
    cont=None,
    use_l1=False, num_workers=32, data_dir=None, save_dir='mount/shared/', **kwargs,
):

    # CONFIG
    energy_loss = get_energy_loss(config=loss_config, mode=mode, **kwargs)

    if data_dir is None:
        buildings = ["almena", "albertville"]
        train_subset_dataset = TaskDataset(buildings, tasks=[tasks.rgb, tasks.normal, tasks.principal_curvature])
    else:
        train_subset_dataset = ImageDataset(data_dir=data_dir)
        data_dir = 'CUSTOM'

    train_subset = RealityTask("train_subset", train_subset_dataset, batch_size=batch_size, shuffle=False)

    if subset_size is None:
        subset_size = len(train_subset_dataset)
    subset_size = min(subset_size, len(train_subset_dataset))

    # GRAPH
    realities = [train_subset]
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

    print('file', cont)
    graph.load_weights(cont)
    graph.compile(optimizer=None)

    # Add consistency links
    graph.edge_map[str(('rgb', 'reshading'))].model.load_weights('./models/rgb2reshading_consistency.pth',backward_compatible=True)
    graph.edge_map[str(('rgb', 'depth_zbuffer'))].model.load_weights('./models/rgb2depth_consistency.pth',backward_compatible=True)
    graph.edge_map[str(('rgb', 'normal'))].model.load_weights('./models/rgb2normal_consistency.pth',backward_compatible=True)


    energy_losses, mse_losses = [], []
    percep_losses = defaultdict(list)

    energy_mean_by_blur, energy_std_by_blur = [], []
    error_mean_by_blur, error_std_by_blur = [], []

    energy_losses, error_losses = [], []

    energy_losses_all, energy_losses_headings = [], []

    fnames = []
    train_subset.reload()
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


    # log losses
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

    z_score = lambda x: (x - x.mean()) / x.std()
    def get_standardized_energy(df, use_std=False, compare_to_in_domain=False):
        percepts = [c for c in df.columns if 'percep' in c]
        stdize = lambda x: (x - x.mean()).abs().mean()
        means = {k: df[k].mean() for k in percepts}
        stds = {k: stdize(df[k]) for k in percepts}
        stdized = {k: (df[k] - means[k])/stds[k] for k in percepts}
        energies = np.stack([v for k, v in stdized.items() if k[-1] == '_' or '__' in k]).mean(0)
        return energies


    if data_dir is 'CUSTOM':
        eng_curr = np.array(energy_losses).mean()
        df = pd.read_csv(os.path.join(save_dir, 'data.csv'))
    else:
        percep_losses = { k: v for k, v in zip(energy_losses_headings, np.concatenate(energy_losses_all, axis=-1))}
        df = pd.DataFrame(both(
                        {'energy': save_energy_losses, 'error': save_error_losses },
                        percep_losses
        ))
        df.to_csv(f"{save_dir}/data.csv", mode='w', header=True)

    # compuate correlation
    df['normalized_energy'] = get_standardized_energy(df, use_std=False)
    df = df[df['normalized_energy'] > -50]
    df['normalized_error'] = z_score(df['error'])
    print(scipy.stats.spearmanr(z_score(df['error']), df['normalized_energy']))
    print("Pearson r:", scipy.stats.pearsonr(df['error'], df['normalized_energy']))

    # plot correlation
    plt.figure(figsize=(4,4))
    g = sns.regplot(df['normalized_error'], df['normalized_energy'],robust=False)
    pdb.set_trace()
    if data_dir is 'CUSTOM':
        ax1 = g.axes
        ax1.axhline(eng_curr, ls='--', color='red')
        ax1.text(0.5, 25, "Query Image Energy Line")
    plt.xlabel('Error (z-score)')
    plt.ylabel('Energy (z-score)')
    plt.title('')
    plt.savefig(f'./energy.pdf')


if __name__ == "__main__":
    Fire(main)
