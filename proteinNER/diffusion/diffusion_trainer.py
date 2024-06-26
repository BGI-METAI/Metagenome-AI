#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : diffusion_trainer.py
# @Author  : zhangchao
# @Date    : 2024/6/26 17:39 
# @Email   : zhangchao5@genomics.cn
import torch
from tqdm import tqdm


def train_one_loop(encoding_model, diffusion, decoding_model, data_loader, max_timestep=1000):
    encoding_model.train()
    batch_iterator = tqdm(
        data_loader,
        desc=''
    )
    for idx, sample in enumerate(batch_iterator):
        input_ids, attention_mask, batch_label = sample
        input_ids = input_ids if isinstance(torch.Tensor, input_ids) else torch.tensor(input_ids)
        attention_mask = attention_mask if isinstance(torch.Tensor, attention_mask) else torch.tensor(attention_mask)
        batch_label = input_ids if isinstance(torch.Tensor, batch_label) else torch.tensor(batch_label)

        timestep = torch.randint(low=1, high=max_timestep, size=input_ids.size(0))

        x0 = encoding_model(input_ids, attention_mask, batch_label)
        xts, gt_noise = diffusion.feed_forward(x0, timestep)

