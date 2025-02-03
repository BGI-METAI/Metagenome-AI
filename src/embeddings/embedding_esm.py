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

import os
import esm
import torch
from torch.nn import Identity
import pickle
import pathlib
import logging

from embeddings.embedding import Embedding


class EsmEmbedding(Embedding):
    def __init__(self, config, pooling="mean"):
        if "model_name_or_path" in config:
            esm_model_path = config["model_name_or_path"]
        else:
            esm_model_path = "esm2_t33_650M_UR50D"
        model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_path)
        # model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        self.model = model
        
        if ('finetune_model_path' in config) and os.path.exists(config['finetune_model_path']):
            self.model.load_state_dict(torch.load(config['finetune_model_path']))
            logging.info(f"Using finetuned model. Path: {config['finetune_model_path']}")

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
        self.target_layer_index = config.get("target_layer_index", None)
        # Ensure the target layer index is within a valid range.
        if self.target_layer_index is not None and not (1 <= self.target_layer_index <= len(self.model.layers)):
            self.target_layer_index = None
        self.hidden_states = {}
        self.hooks = []
        for idx, layer in enumerate(self.model.layers, start=1):
            if self.target_layer_index is not None and idx == self.target_layer_index:
                # Register hook only if target_layer_index is not None and matches the layer
                self.hooks.append(layer.register_forward_hook(self._save_hidden_states))

    def _save_hidden_states(self, module, input, output):
        """Hook function to capture hidden states for the target layer."""
        # Clear previous hidden states to store only the latest output
        if self.hidden_states:
            self.hidden_states.clear()
        layer_output = output[0]  # Get the output with shape [batch_size, seq_length, embedding_dim]
        layer_key = f"layer{self.target_layer_index}"
        if layer_key not in self.hidden_states:
            self.hidden_states[layer_key] = []
        # Store the detached output
        self.hidden_states[layer_key].append(layer_output.detach().cpu().numpy())

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
        return self._pooling(
            self.pooling, esm_result["logits"], batch_tokens, self.alphabet.padding_idx
        )

    def to(self, device):
        self.model = self.model.to(device)

    def get_embedding_dim(self):
        return self.model.embed_dim

    def _pooling(self, strategy, tensors, batch_tokens, pad_token_id):
        """Perform pooling on [batch_size, seq_len, emb_dim] tensor

        Args:
            strategy: One of the values ["mean", "max", "cls"]
        """
        if strategy == "cls":
            seq_repr = tensors[:, 0, :]
        elif strategy == "mean":
            seq_repr = []
            batch_lens = (batch_tokens != pad_token_id).sum(1)

            for i, tokens_len in enumerate(batch_lens):
                seq_repr.append(tensors[i, 1 : tokens_len - 1].mean(0))

            seq_repr = torch.vstack(seq_repr)
        else:
            raise NotImplementedError("This type of pooling is not supported")
        return seq_repr

    def store_embeddings(self, batch, out_dir, target_layer_index):
        """Store each protein embedding in a separate file named [protein_id].pkl

        Save all types of poolings such that each file has a [3, emb_dim]
        where rows 0, 1, 2 are mean, max, cls pooled respectively

        Args:
            batch: Each sample contains protein_id and sequence
        """
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        data = [
            (protein_id, seq)
            for protein_id, seq in zip(batch["protein_id"], batch["sequence"])
        ]
        batch_labels, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            esm_result = self.model(batch_tokens)

        esm_result = esm_result["logits"].detach().cpu()
        mean_max_cls_embeddings = []
        mean_embeddings = self._pooling(
            "mean", esm_result, batch_tokens, self.alphabet.padding_idx
        )
        cls_embeddings = self._pooling(
            "cls", esm_result, batch_tokens, self.alphabet.padding_idx
        )
        mean_target_layer_embeddings = None
        if f"layer{target_layer_index}" in self.hidden_states:
            target_layer_embeddings = torch.tensor(self.hidden_states[f"layer{target_layer_index}"][0]).permute(1, 0, 2)
            mean_target_layer_embeddings = self._pooling(
                    "mean", target_layer_embeddings, batch_tokens, self.alphabet.padding_idx
            )
        for i, (protein_id, mean_emb, cls_emb) in enumerate(zip(
            batch_labels, mean_embeddings, cls_embeddings
        )):
            embeddings_dict = {
                "mean": mean_emb.numpy(),
                "cls": cls_emb.numpy(),
            }
            if mean_target_layer_embeddings is not None:
                embeddings_dict[f"mean_hidden_layer{target_layer_index}"] = mean_target_layer_embeddings[i].numpy()
            with open(f"{out_dir}/{protein_id}.pkl", "wb") as file:
                pickle.dump(embeddings_dict, file)
