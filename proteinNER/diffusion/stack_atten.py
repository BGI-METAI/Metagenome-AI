#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : stack_atten.py
# @Author  : zhangchao
# @Date    : 2024/6/25 16:03 
# @Email   : zhangchao5@genomics.cn
import math
import torch
import torch.nn as nn


def positional_timestep(timestep, dim, max_period=10000):
    half = dim // 2
    freq = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timestep.device)
    args = timestep[:, None].float() * freq[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CustomAttention(nn.Module):
    def __init__(self, d_model, n_header):
        super(CustomAttention, self).__init__()
        assert d_model % n_header == 0
        self.d_header = d_model // n_header
        self.n_header = n_header
        self.d_model = d_model
        self.w_qkv = nn.Linear(d_model, 3 * d_model)
        self.drop = nn.Dropout(p=0.1)
        self.fc_out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.w_qkv(x)
        query, key, value = torch.split(x, self.d_model, -1)
        query = query.view(batch_size, -1, self.n_header, self.d_header)
        key = key.view(batch_size, -1, self.n_header, self.d_header)
        value = value.view(batch_size, -1, self.n_header, self.d_header)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_header)
        scores = torch.nn.functional.softmax(scores, dim=-1)
        scores = self.drop(scores)

        x = torch.matmul(scores, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_header * self.d_header)

        return self.fc_out(x)


class CustomSelfAttn(nn.Module):
    def __init__(self, d_model, n_header):
        super(CustomSelfAttn, self).__init__()
        self.attn = CustomAttention(d_model, n_header)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model)
        )
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        attn = self.attn(x)
        x = self.norm1(x + attn)
        feed = self.feed_layer(x)
        return self.drop(self.norm2(x + feed))


class ProteinFuncAttention(nn.Module):
    def __init__(self, d_model, n_header):
        super(ProteinFuncAttention, self).__init__()
        self.d_model = d_model

        self.timestep_transition_layer = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ly_norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(p=0.1)

        self.input_project_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        self.ly_norm2 = nn.LayerNorm(d_model)

        self.stack_attn = nn.ModuleList()
        for _ in range(8):
            self.stack_attn.append(
                CustomSelfAttn(d_model, n_header)
            )

        self.output_project_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model)
        )
        self.ly_norm3 = nn.LayerNorm(d_model)

    def forward(self, x, timestep):
        timestep_embed = self.timestep_transition_layer(positional_timestep(timestep, self.d_model))
        timestep_embed = self.drop1(self.ly_norm1(timestep_embed))

        x_embed = self.input_project_layer(x)

        x_embed = self.ly_norm2(x_embed + timestep_embed.unsqueeze(1).expand(-1, x.size(1), -1))
        for idx, ly in enumerate(self.stack_attn):
            x_embed = ly(x_embed)

        out = self.output_project_layer(x_embed)
        out = self.ly_norm3(out)
        return out
