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
            esm_result = self.model(batch_tokens)

        # The first token of every sequence is always a special classification token ([CLS]).
        # The final hidden state corresponding to this token is used as the aggregate sequence representation
        # for classification tasks.
        return self._pooling(self.pooling, esm_result["logits"], batch_tokens)

    def to(self, device):
        self.model = self.model.to(device)

    def get_embedding_dim(self):
        return self.model.embed_dim

    def _pooling(self, strategy, tensors, batch_tokens):
        """Perform pooling on [batch_size, seq_len, emb_dim] tensor

        Args:
            strategy: One of the values ["mean", "max", "cls"]
        """
        if strategy == "cls":
            seq_repr = tensors[:, 0, :]
        elif strategy == "mean":
            seq_repr = []
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(tensors[i, 1 : tokens_len - 1].mean(0))

            seq_repr = torch.vstack(seq_repr)
        elif strategy == "max":
            seq_repr = []
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(tensors[i, 1 : tokens_len - 1].max(0))

            seq_repr = torch.vstack(seq_repr)
        else:
            raise NotImplementedError("This type of pooling is not supported")
        return seq_repr

    def store_embeddings(self, batch):
        """Store each protein embedding in a separate file named [protein_id].pkl

        Save all types of poolings such that each file has a [3, emb_dim]
        where rows 0, 1, 2 are mean, max, cls pooled respectively

        Args:
            batch: Each sample contains protein_id and sequence
        """
        data = [
            (protein_id, seq)
            for protein_id, seq in zip(batch["protein_id"], batch["sequence"])
        ]
        batch_labels, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            esm_result = self.model(batch_tokens).detach().cpu()

        mean_max_cls_embeddings = []
        mean_embeddings = self._pooling("mean", esm_result["logits"], batch_tokens)
        max_embeddings = self._pooling("max", esm_result["logits"], batch_tokens)
        cls_embeddings = self._pooling("cls", esm_result["logits"], batch_tokens)

        # actually easier to create a .pkl dict with keys mean, max, cls, seq, labels, seq_len
        # TODO change this and do save by the protein ID
        for protein_id, mean_emb, max_emb, cls_emb in zip(batch_labels, mean_embeddings, max_embeddings, cls_embeddings):
            embeddings = torch.vstack([mean_embeddings, max_embeddings, cls_embeddings])



