
import os, sys, math, random, itertools, heapq
from collections import namedtuple, defaultdict
from functools import partial, reduce
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from models import TrainableModel, WrapperModel
from datasets import TaskDataset
from task_configs import get_task, task_map, tasks, get_model, RealityTask
from transfers import Transfer, RealityTransfer, get_named_transfer
import transforms


class TaskGraph(TrainableModel):
    """Basic graph that encapsulates set of edge constraints. Can be saved and loaded
    from directories."""

    def __init__(self, 
            tasks=tasks, edges=None, edges_exclude=None, batch_size=64, reality=None,
            task_filter=[tasks.segment_semantic, tasks.class_scene],
            anchored_tasks=[],
            initialize_first_order=True,
        ):
        super().__init__()
        self.tasks = list(set(tasks) - set(task_filter))
        self.edges, self.adj, self.in_adj = [], defaultdict(list), defaultdict(list)
        self.edge_map = {}
        self.reality=reality
        self.anchored_tasks = set(anchored_tasks)

        # construct transfer graph
        for src_task, dest_task in itertools.product(self.tasks, self.tasks):
            key = (src_task.name, dest_task.name)
            if edges is not None and key not in edges: continue
            # if edges_exclude is not None and key in edges_exclude: continue
            if isinstance(src_task, RealityTask):
                if dest_task not in src_task.tasks: continue
                if dest_task not in anchored_tasks: continue
                transfer = RealityTransfer(src_task, dest_task)
                self.edges += [transfer]
                self.adj[src_task] += [transfer]
                self.in_adj[dest_task] += [transfer]
                continue

            if isinstance(dest_task, RealityTask): continue
            if src_task == dest_task: continue

            transfer = get_named_transfer(Transfer(src_task, dest_task))
            if transfer.model_type is None: 
                print ("Failed: ", transfer)
                continue
            self.edges += [transfer]
            self.adj[src_task] += [transfer]
            self.in_adj[dest_task] += [transfer]
            self.edge_map[key] = transfer

        self.initialize_first_order = initialize_first_order
        self.batch_size = batch_size
        if batch_size > 0: 
            self.init_params()

    def init_params(self):
        self.estimates = WrapperModel(
            nn.ParameterDict({
                task.name: nn.Parameter(torch.randn(
                        *([self.batch_size] + list(task.shape))
                    ).requires_grad_(True).to(DEVICE)*task.variance
                ) for task in self.tasks
            })
        )

        for task in self.anchored_tasks:
            if task is self.reality: continue
            self.estimates[task.name].data = self.reality.task_data[task].to(DEVICE)

        
        if self.initialize_first_order:
            for dest_task in self.tasks:
                found_src = False
                for src_task in self.anchored_tasks:
                    for transfer in self.adj[src_task]:
                        if transfer.dest_task == dest_task:
                            self.estimates[dest_task.name].data = transfer(self.estimates[src_task.name]).data
                            found_src = True
                            break

                    if found_src: break
        self.init = {task.name:torch.tensor(self.estimates[task.name]).cpu() for task in self.tasks}
        tasks_theta = list(set(self.tasks) - self.anchored_tasks)
        self.p = WrapperModel(
            nn.ParameterDict({
                task.name: nn.Parameter(
                    torch.tensor([2.0, 8.0]).requires_grad_(True).to(DEVICE)
                ) for task in tasks_theta
            })
        )

    def prob(self, task):
        if task in self.anchored_tasks:
            return torch.tensor(1.0, device=DEVICE)
        return F.softmax(F.relu(self.p[task.name]), dim=0)[1]
    
    def dist(self, task):
        if task in self.anchored_tasks:
            return torch.tensor([0.0, 1.0], device=DEVICE)
        return F.softmax(self.p[task.name].clamp(min=1e-3, max=1-1e-3), dim=0)

    def estimate(self, task):
        x = self.estimates[task.name]
        return x.detach() if task in self.anchored_tasks else x

    # def conditional(self, transfer):

    #     A, B = transfer.src_task, transfer.dest_task

    #     Ax = self.estimates[A.name]
    #     Ax = Ax.detach() if A in self.anchored_tasks else Ax
    #     Bx = self.estimates[B.name]
    #     Bx = Bx.detach() if B in self.anchored_tasks else Bx

    #     est_mse = transfer.dest_task.norm(Bx, transfer(Ax))[0]
    #     gt_mse = B.variance*(1 - self.prob(B))
    #     mse = (est_mse**2 + gt_mse**2)**(0.5)
    #     # pdf is uniform in angle?
    #     # Do some clever math here to ensure nothing becomes 0.0 for no reason
    #     # Without causing pathological edge cases you fuck
    #     # Lower bound is gt_mse - est_mse, higher bound is gt_mse + est_mse
    #     # 
    #     # 
    #     p = F.relu(1 - mse/B.variance)/self.prob(A)
    #     # if p.data.cpu().numpy().mean() <= 1e-3:
    #     #     IPython.embed()
    #     p = p.clamp(min=1e-3, max=1-1e-3)
    #     prob = torch.stack([1-p, p])

    #     divergence = F.kl_div(torch.log(prob), self.dist(B))
    #     print (f"{transfer}: p(F|E)={p} -> p(F)={self.prob(B)} ... transfer_mse={est_mse}, variance={B.variance}, divergence={divergence}")
    #     # divergence.backward(retain_graph=True)

    #     return prob

    # def free_energy(self, sample=10, tve=False):

    #     def tve_loss(x):
    #         dx = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).mean()
    #         dy = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).mean()
    #         return (dx+dy)/2
    #     def norm(transfer):
    #         A, B = transfer.src_task, transfer.dest_task
    #         Ax = self.estimates[A.name]
    #         Ax = Ax.detach() if A in self.anchored_tasks else Ax
    #         Bx = self.estimates[B.name]
    #         Bx = Bx.detach() if B in self.anchored_tasks else Bx

    #         tve = 0 if tve and A in self.anchored_tasks else tve_loss(Ax)
    #         return (transfer.dest_task.norm(Bx, transfer(Ax))[0]/B.variance) + tve

    #     task_data = [
    #         norm(transfer) for transfer in random.sample(self.edges, sample)
    #     ]
    #     return sum(task_data)/len(task_data)


    def plot_paths(self, logger, src_tasks=None, dest_tasks=[tasks.normal], show_images=False, max_len=4):

        src_tasks = src_tasks or self.anchored_tasks
        with torch.no_grad():
            def dfs(task, X, max_len=max_len, history=[]):
                if task in dest_tasks: yield (history, X)
                if isinstance(task, RealityTask) or max_len == 0: return

                for transfer in self.adj[task]:
                    yield from dfs(
                        transfer.dest_task, transfer(X), max_len=max_len-1, 
                        history=history+[transfer]
                    )

            self.mse, self.pathnames, images = defaultdict(list), defaultdict(list), defaultdict(list)
            for history, Y in itertools.chain.from_iterable(
                dfs(src_task, self.estimates[src_task.name], history=[src_task]) \
                for src_task in src_tasks
            ):
                print ("DFS path: ", history)
                if len(history) == 1:
                    continue
                dest_task = history[-1].dest_task
                pathname = reduce(lambda a, b: f"{b}({a})", history)
                mse = dest_task.norm(self.reality.task_data[dest_task].to(DEVICE), Y)[0].data.cpu().numpy().mean()
                print (mse)

                self.mse[dest_task] += [mse]
                self.pathnames[dest_task] += [pathname]
                images[dest_task] += [(-mse, pathname, Y.detach())]

            if show_images:
                for task in dest_tasks:
                    for mse, pathname, Y in heapq.nlargest(5, images[task]):
                        logger.images(Y, pathname, resize=256)
                    logger.images(self.reality.task_data[task].to(DEVICE), f"GT {task}", resize=256)

        self.update_paths(logger)

    def update_paths(self, logger):

        with torch.no_grad():
            for task in self.pathnames:
                curr_mse = task.norm(self.reality.task_data[task].to(DEVICE).detach(), 
                    self.estimates[task.name])[0].data.cpu().numpy().mean()
                data, rownames = zip(*sorted(zip(self.mse[task],  self.pathnames[task])))
                logger.bar([curr_mse] + list(data), f'{task}_path_mse', opts={'rownames': ["current"] + list(rownames)})

    def plot_estimates(self, logger, suffix=""):
        for task in self.tasks:
            if task in self.anchored_tasks:
                task.plot_func(self.estimates[task.name], task.name + suffix, logger, nrow=1)
                continue
            grouped = [self.estimates[task.name].cpu(), self.init[task.name].cpu()]
            if task in self.reality.task_data:
                grouped.append(self.reality.task_data[task].cpu())
            interleave = torch.stack([y for x in zip(*grouped) for y in x])
            task.plot_func(interleave, task.name + suffix, logger, nrow=len(grouped))

    # def conditional(self, transfer):

    #     A, B = transfer.src_task, transfer.dest_task

    #     Ax = self.estimate(A)
    #     Bx = self.estimate(B)

    #     est_mse = transfer.dest_task.norm(Bx, transfer(Ax))[0]
    #     gt_mse = B.variance*(1 - self.prob(B))
    #     mse = (est_mse**2 + gt_mse**2)**(0.5)
        
    #     # pdf is uniform in angle?
    #     # Do some clever math here to ensure nothing becomes 0.0 for no reason
    #     # Without causing pathological edge cases

    #     p = F.relu(1 - mse/B.variance)/self.prob(A)
    #     p = p.clamp(min=1e-3, max=1-1e-3)
    #     prob = torch.stack([1-p, p])

    #     divergence = F.kl_div(torch.log(prob), self.dist(B))
    #     print (f"{transfer}: p(F|E)={p} -> p(F)={self.prob(B)} ... transfer_mse={est_mse}, variance={B.variance}, divergence={divergence}")

    #     return prob

    def free_energy(self, sample=10):

        def norm(transfer):
            return (
                transfer.dest_task.norm(
                    self.estimate(transfer.dest_task), 
                    transfer(self.estimate(transfer.src_task))
                )[0]/(transfer.dest_task.variance**(0.5))
            )

        task_data = [
            norm(transfer) for transfer in random.sample(self.edges, sample)
        ]
        return sum(task_data)/len(task_data)

    def averaging_step(self, sample=10):
        for task in self.tasks:
            if isinstance(task, RealityTask): continue
            estimates = (transfer(self.estimate(transfer.src_task)) for transfer in self.in_adj[task])
            average = sum(estimates)/len(self.in_adj[task])
            self.estimates[task.name].data = average.data
            # self.estimates[task.name].data = (self.estimates[task.name].data + average.data)/2.0     

    def incoming_transfers(self, task):
        images, task_names = [], []
        for transfer in self.edges:
            if transfer.dest_task.name != task.name: continue
            images.append(transfer(self.estimate(transfer.src_task)))
            task_names.append(transfer.src_task.name)
        return torch.stack(images), task_names

    def cycle_loss_test(self):

        f = self.edge_map[('normal', 'principal_curvature')]
        F = self.edge_map[('principal_curvature', 'normal')]
        y_hat = self.reality.task_data[tasks.normal].to(DEVICE).detach()
        y = self.estimate(tasks.normal)
        f_y, f_y_hat = f(y), f(y_hat)
        curv_loss, _ = tasks.principal_curvature.norm(f_y, f_y_hat)
        cycle_loss, _ = tasks.normal.norm(F(f_y), y)
        return curv_loss + cycle_loss, (curv_loss.detach(), cycle_loss.detach())

    def cycle_synthesis_test(self):

        f = self.edge_map[('normal', 'principal_curvature')]
        F = self.edge_map[('principal_curvature', 'normal')]
        y_hat = self.reality.task_data[tasks.normal].to(DEVICE).detach()

        y = self.estimate(tasks.normal)
        z = self.estimate(tasks.principal_curvature)
        f_y, f_y_hat = f(y), f(y_hat)

        # f(y) -> f(y_hat)
        # z -> f(y)
        # F(z) -> y
        
        curv_loss, _ = tasks.principal_curvature.norm(f_y, f_y_hat)
        cycle_step1, _ = tasks.principal_curvature.norm(f_y, z)
        cycle_step2, _ = tasks.normal.norm(F(z), y)
        normal_error, _ = tasks.normal.norm(y, y_hat)
        return curv_loss + cycle_step1 + cycle_step2, (curv_loss.detach(), cycle_step1.detach(), cycle_step2.detach(), normal_error.detach())


    # def free_energy(self, sample=12):

    #     losses = []
    #     for transfer in self.edges:
    #         if transfer.src_task in [tasks.normal, tasks.sobel_edges, tasks.reshading, tasks.edge_occlusion]:

    #             losses.append(transfer.dest_task.norm( 
    #                 self.estimates[transfer.dest_task.name].detach(),
    #                 transfer(self.estimates[transfer.src_task.name])
    #             )[0]/transfer.dest_task.variance)

    #         if transfer.dest_task in [tasks.normal, tasks.sobel_edges, tasks.reshading, tasks.edge_occlusion]:

    #             losses.append(transfer.dest_task.norm( 
    #                 self.estimates[transfer.dest_task.name],
    #                 transfer(self.estimates[transfer.src_task.name]).detach()
    #             )[0]/transfer.dest_task.variance)
        
    #     return sum(losses)
    
    # def free_energy(self, sample=10):
    #     task_data = [
    #         F.kl_div( 
    #             torch.log(self.conditional(transfer)),
    #             self.dist(transfer.dest_task)
    #         ) for transfer in random.sample(self.edges, sample)
    #     ]

    #     return sum(task_data)/len(task_data)



if __name__ == "__main__":

    reality = RealityTask('albertville', 
        dataset=TaskDataset(
            buildings=['albertville'],
            tasks=[tasks.rgb, tasks.normal],
        ),
        tasks=[tasks.rgb, tasks.normal],
        batch_size=64
    )

    graph = TaskGraph(
        tasks=[reality, tasks.rgb, tasks.normal, tasks.principal_curvature, tasks.depth_zbuffer]
    )
    print (graph.edges)
    
