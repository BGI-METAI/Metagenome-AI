#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : denoise_model.py
# @Author  : zhangchao
# @Date    : 2024/7/5 11:46 
# @Email   : zhangchao5@genomics.cn
import math
import torch
import torch.nn as nn
from transformers import T5EncoderModel
from transformers.models.t5.modeling_t5 import T5LayerNorm
from peft import LoraConfig, get_peft_model


class DenoiseModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_classes,
            *,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        super(DenoiseModel, self).__init__()
        self.decoder = T5EncoderModel.from_pretrained(model_name_or_path)
        d_model = self.decoder.config.d_model
        peft_config = LoraConfig(
            inference_mode=inference_mode,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['q', 'k', 'v', 'o']
        )
        self.decoder = get_peft_model(self.decoder.encoder.block, peft_config)

        self.layer_norm = T5LayerNorm(d_model)
        self.drop = nn.Dropout(p=0.1)
        self.final_classifier = nn.Linear(
            in_features=d_model,
            out_features=num_classes,
            bias=False
        )

        self.timestep_transition = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, x, timestep):
        timestep_embed = self.timestep_transition(timestep)
        x += timestep_embed.unsqueeze(1).expand(-1, x.size(1), -1)
        x = self.decoder(x)
        x = self.layer_norm(x)
        x = self.drop(x)
        x = self.final_classifier(x)
        return x

    def positional_timestep(self, timestep, dim, max_period=10000):
        half = dim // 2
        freq = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=timestep.device)
        args = timestep[:, None].float() * freq[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
