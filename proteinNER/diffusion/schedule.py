#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : schedule.py
# @Author  : zhangchao
# @Date    : 2024/6/25 14:48 
# @Email   : zhangchao5@genomics.cn
import numpy as np
import math
import torch


class BetaSchedule:

    def __init__(self, num_timestep):
        self.num_timestep = num_timestep

    @staticmethod
    def betas4alpha_bar(timestep, alpha_bar, max_beta=0.999):
        """
        Create a beta schedule that discretizes the given alpha_t_bar function,
        which defines the cumulative product of (1-beta) over time from t = [0,1].

        Args:
            timestep: the number of betas to produce.
            alpha_bar: a lambda that takes an argument t from 0 to 1 and
                       produces the cumulative product of (1-beta) up to that
                       part of the diffusion process.
            max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
        """
        betas = []
        for i in range(timestep):
            t1 = i / timestep
            t2 = (i + 1) / timestep
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas, dtype=np.float64)

    def cosine_beta_schedule(self):
        return self.betas4alpha_bar(
            timestep=self.num_timestep,
            alpha_bar=lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        )

    def linear_beta_schedule(self):
        scale = 1000 / self.num_timestep
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, self.num_timestep, dtype=np.float64)


class TimestepScheduleSampler:
    def __init__(self, diffusion):
        self.diffusion = diffusion

    def weights(self):
        return np.ones([self.diffusion.num_timesteps])

    def sample(self, batch_size, device):
        """
        Importance-sample timestep for a batch.

        Args:
            batch_size: the number of timestep.
            device: the torch device to save to.

        Returns:
            a tuple (timestep, weight):
            - timestep: a tensor of timestep indices.
            - weight: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights
