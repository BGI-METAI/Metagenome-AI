#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 9:47
# @Author  : zhangchao
# @File    : model.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import T5EncoderModel, T5Model


class ProtTransT5EmbeddingPEFTModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        super(ProtTransT5EmbeddingPEFTModel, self).__init__()
        self.base_model = T5EncoderModel.from_pretrained(model_name_or_path)
        peft_config = LoraConfig(
            inference_mode=lora_inference_mode,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.lora_embedding = get_peft_model(self.base_model, peft_config)

    def forward(self, input_ids, attention_mask):
        return self.lora_embedding(input_ids, attention_mask).last_hidden_state


class ProtTransT5ForAAClassifier(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_classes,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
    ):
        super(ProtTransT5ForAAClassifier, self).__init__()
        self.embedding = ProtTransT5EmbeddingPEFTModel(
            model_name_or_path=model_name_or_path,
            lora_inference_mode=lora_inference_mode,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.classifier = nn.Linear(self.embedding.lora_embedding.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.embedding(input_ids, attention_mask))


class ProtTransT5MaskPEFTModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_classes,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
    ):
        super(ProtTransT5MaskPEFTModel, self).__init__()
        self.embedding = ProtTransT5EmbeddingPEFTModel(
            model_name_or_path=model_name_or_path,
            lora_inference_mode=lora_inference_mode,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.classifier = nn.Linear(self.embedding.lora_embedding.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.embedding(input_ids, attention_mask))
