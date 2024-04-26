#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   embedding_esm.py
@Time    :   2024/04/26 14:48:34
@Author  :   Nikola Milicevic
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

import esm
import torch
from torch.nn import Identity

from embedding import Embedding


class EsmEmbedding(Embedding):
    def __init__(self, pooling="mean"):
        # model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        # model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.model = model
        self.alphabet = alphabet
        self.model.contact_head = Identity()
        self.model.emb_layer_norm_after = Identity()
        self.model.lm_head = Identity()
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.embed_dim = model.embed_dim
        self.pooling = pooling
        # Freeze the parameters of ESM
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, batch):
        data = [
            (target, seq) for target, seq in zip(batch["target"], batch["sequence"])
        ]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            res = self.model(batch_tokens)

        # The first token of every sequence is always a special classification token ([CLS]).
        # The final hidden state corresponding to this token is used as the aggregate sequence representation
        # for classification tasks.
        if self.pooling == "cls":
            seq_repr = res["logits"][:, 0, :]
        elif self.pooling == "mean":
            seq_repr = []
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(res["logits"][i, 1 : tokens_len - 1].mean(0))

            seq_repr = torch.vstack(seq_repr)
        elif self.pooling == "max":
            seq_repr = []
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(res["logits"][i, 1 : tokens_len - 1].max(0))

            seq_repr = torch.vstack(seq_repr)
        else:
            raise NotImplementedError()
        return seq_repr

    def to(self, device):
        self.model = self.model.to(device)

    def get_embedding_dim(self):
        return self.model.embed_dim
