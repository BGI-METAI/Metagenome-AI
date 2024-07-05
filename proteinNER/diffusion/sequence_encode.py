#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : sequence_encode.py
# @Author  : zhangchao
# @Date    : 2024/6/26 9:53 
# @Email   : zhangchao5@genomics.cn
import math
import torch
import torch.nn as nn
from transformers import T5EncoderModel
from transformers import AutoModel
from peft import LoraConfig, get_peft_model


# class SequenceLabelEmbedding(nn.Module):
#     def __init__(self, num_embeddings, embedding_dim=1024):
#         super(SequenceLabelEmbedding, self).__init__()
#         self.share_embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
#         self.net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=12)
#
#     def forward(self, x):
#         x = self.share_embed(x)
#         B, L, C = x.size()
#         pe = torch.zeros(L, C, device=x.device, dtype=torch.float)
#         position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         x += pe.unsqueeze(0)
#         x = self.net(x)
#         return x


class ProteinSequenceEmbedding(nn.Module):
    def __init__(self, model_name_or_path):
        super(ProteinSequenceEmbedding, self).__init__()
        self.embedding = T5EncoderModel.from_pretrained(model_name_or_path)
        for k, v in self.embedding.named_parameters():
            v.requires_grad = False

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            embedding = self.embedding(input_ids, attention_mask).last_hidden_state
        return embedding


class ProteinLabelEmbedding(nn.Module):
    def __init__(
            self,
            pretrained_model_name_or_path,
            num_vocabs,
            *,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        super(ProteinLabelEmbedding, self).__init__()
        self.base_embedding = AutoModel.from_pretrained(pretrained_model_name_or_path)
        self.base_embedding.embeddings.word_embeddings = nn.Embedding(num_vocabs, embedding_dim=1024)
        lora_config = LoraConfig(
            inference_mode=inference_mode,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=['query', 'key', 'value', 'dense']
        )
        self.base_embedding = get_peft_model(self.base_embedding, lora_config)

    def forward(self, input_ids, attention_mask):
        return self.base_embedding(input_ids, attention_mask)


class SequenceEncoder(nn.Module):
    def __init__(
            self,
            pretrained_sequence_embed_model_name_or_path,
            label_embed_model_name_or_path,
            num_vocabs
    ):
        super(SequenceEncoder, self).__init__()
        self.sequence_model = ProteinSequenceEmbedding(pretrained_sequence_embed_model_name_or_path)
        self.label_model = ProteinLabelEmbedding(label_embed_model_name_or_path, num_vocabs=num_vocabs)

    def forward(
            self,
            seq_input_ids,
            seq_attention_mask,
            raw_label_input_ids,
            raw_label_attention_mask,
            mask_label_input_ids,
            mask_label_attention_mask
    ):
        seq_embedding = self.sequence_model(seq_input_ids, seq_attention_mask)
        raw_embedding = self.label_model(raw_label_input_ids, raw_label_attention_mask)
        mask_embedding = self.label_model(mask_label_input_ids, mask_label_attention_mask)
        return seq_embedding, raw_embedding, mask_embedding
