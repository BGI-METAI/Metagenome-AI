#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 10:53
# @Author  : zhangchao
# @File    : finetuning_prottrans.py
# @Email   : zhangchao5@genomics.cn

import torch.nn as nn
from transformers import T5EncoderModel
from peft import LoraConfig, get_peft_model


class FineTuneProtTransAAModel(nn.Module):
    def __init__(self, model_name_or_path, n_classes):
        super(FineTuneProtTransAAModel, self).__init__()
        embedding = T5EncoderModel.from_pretrained(model_name_or_path)
        peft_config = LoraConfig(inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
        self.embedding = get_peft_model(embedding, peft_config)
        self.classifier = nn.Linear(self.embedding.config.d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.classifier(x)
        return x

