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


class SequenceLabelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim=1024):
        super(SequenceLabelEmbedding, self).__init__()
        self.share_embed = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True)
        self.net = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=12)

    def forward(self, x):
        x = self.share_embed(x)
        B, L, C = x.size()
        pe = torch.zeros(L, C, device=x.device, dtype=torch.float)
        position = torch.arange(0, L, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        x += pe.unsqueeze(0)
        x = self.net(x)
        return x


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


# class ProteinSeqEncoder(nn.Module):
#     def __init__(self, model_name_or_path, num_labels, label_embedding_dims=1024):
#         super().__init__()
#         self.sequence_embedding = ProteinSequenceEmbedding(model_name_or_path)
#         self.label_embedding = SequenceLabelEmbedding(num_embeddings=num_labels, embedding_dim=label_embedding_dims)
#
#     def forward(self, input_ids, attention_mask, labels):
#         seq_emd = self.sequence_embedding(input_ids, attention_mask)
#         lab_emd = self.label_embedding(labels)
#         return torch.concatenate((seq_emd, lab_emd), dim=0)



