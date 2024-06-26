#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : diffusion.py
# @Author  : zhangchao
# @Date    : 2024/6/26 18:02 
# @Email   : zhangchao5@genomics.cn
import torch
import numpy as np


class SimpleDiffusion:
    def __init__(self, betas):
        betas = betas if isinstance(torch.Tensor, betas) else torch.tensor(betas)
        assert len(betas.size()) == 1, 'Betas must be 1-D'
        assert torch.all(betas > 0) and torch.all(betas <= 1)

        self.num_timestep = betas.size(0)

        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], device=self.alphas_cumprod.devive), self.alphas_cumprod[:-1]))
        self.alphas_cumprod_next = torch.cat(
            (self.alphas_cumprod[1:], torch.tensor([0.], device=self.alphas_cumprod.devive)))
        assert self.alphas_cumprod_prev.size(0) == self.num_timestep

        # calculates for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_one_minus_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)



