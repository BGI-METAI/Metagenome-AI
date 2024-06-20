#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : diffusion_model.py
# @Author  : zhangchao
# @Date    : 2024/6/17 16:02 
# @Email   : zhangchao5@genomics.cn
import math
import torch
from dataclasses import field

import numpy as np


class Diffusion:
    alpha: torch.Tensor = field(default=None, metadata={'help': 'diffusion alpha schedule'})
    beta: torch.Tensor = field(default=None, metadata={'help': 'diffusion beta schedule'})

    def __init__(self, **kwargs):
        timestep = kwargs.get('timestep')
        self.initialize(timestep=timestep)

    @staticmethod
    def cosine_beta_schedule(timestep, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timestep + 1
        x = torch.linspace(0, timestep, steps)
        alphas_cumprod = torch.cos(((x / timestep) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def initialize(self, timestep):
        self.beta = self.cosine_beta_schedule(timestep=timestep)
        self.alpha = 1 - self.beta

        self.alpha_cumulative = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_cumulative = torch.sqrt(self.alpha_cumulative)
        self.one_by_sqrt_alpha = 1. / torch.sqrt(self.alpha)
        self.sqrt_one_minus_alpha_cumulative = torch.sqrt(1 - self.alpha_cumulative)




