#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 2:18 PM
# @Author  : zhangchao
# @File    : esm2_embeddings.py
# @Email   : zhangchao5@genomics.cn
import esm
import torch
from torch.nn import Identity

from framework.embeddings import Embeddings


class Esm2Embeddings(Embeddings):
    def __init__(self):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

        self.model = model
        self.alphabet = alphabet
        self.model.contact_head = Identity()
        self.model.emb_layer_norm_after = Identity()
        self.model.lm_head = Identity()
        self.model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        self.embed_dim = model.embed_dim

        # Freeze the parameters of ESM
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, batch, pooling="cls"):
        raise NotImplementedError

    def to(self, device):
        self.model = self.model.to(device)

    def get_embedding_dim(self):
        return self.model.embed_dim
