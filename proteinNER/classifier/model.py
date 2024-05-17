#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 9:47
# @Author  : zhangchao
# @File    : model.py
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn
from torchcrf import CRF
from peft import LoraConfig, get_peft_model
from transformers import T5EncoderModel


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


class ProtT5Conv1dCRF4AAClassifier(nn.Module):
    def __init__(self, model_name_or_path, num_classes):
        super(ProtT5Conv1dCRF4AAClassifier, self).__init__()
        self.base_embedding = T5EncoderModel.from_pretrained(model_name_or_path)
        for k, v in self.base_embedding.named_parameters():
            v.requires_grad = False

        self.transition = nn.Conv1d(
            in_channels=self.base_embedding.config.d_model,
            out_channels=num_classes,
            kernel_size=(3,),
            stride=(1,),
            padding='same'
        )
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            embeddings = self.base_embedding(input_ids, attention_mask).last_hidden_state
        embeddings = self.transition(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        loss = -1 * self.crf(embeddings, labels, mask=attention_mask.byte())
        return loss

    @torch.no_grad()
    def inference(self, input_ids, attention_mask):
        embeddings = self.base_embedding(input_ids, attention_mask).last_hidden_state
        embeddings = self.transition(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        predict = self.crf.decode(embeddings, attention_mask.byte())
        return predict
