#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   embedding_esm3.py
@Time    :   2024/07/10 
@Author  :   Vladimir Kovacevic
@Version :   1.0
@Contact :   vladimirkovacevic@genomics.cn
@Desc    :   None
"""

import pathlib
import pickle

import torch
from torch.nn import Identity
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from esm.pretrained import ESM3_sm_open_v0
from huggingface_hub.hf_api import HfFolder

from embedding_esm import EsmEmbedding

class Esm3Embedding(EsmEmbedding):
    def __init__(self, pooling="mean"):
        login(token="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")
        # HfFolder.save_token(access_token)

        # This will download the model weights and instantiate the model on your machine.
        # model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"
        model = ESM3_sm_open_v0("cuda")

        self.model = model
        self.alphabet = EsmSequenceTokenizer()  # model.get_structure_token_encoder()  #
        # self.model.structure_encoder = Identity()
        self.model.eval()
        self.embed_dim = 1536 # TODO: model.embed_dim
        self.pooling = pooling
        # Freeze the parameters of ESM
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_embedding(self, batch):

        tokens = self.alphabet.batch_encode_plus(batch["sequence"], padding=True)['input_ids']  # encode
        batch_tokens = torch.tensor(tokens, dtype=torch.int64).cuda()  # To GPU

        with torch.no_grad():
            esm_result = self.model(sequence_tokens=batch_tokens)

        return self._pooling(self.pooling, esm_result.embeddings, batch_tokens, self.alphabet.pad_token_id)
    
    def store_embeddings(self, batch, out_dir):
        """Store each protein embedding in a separate file named [protein_id].pkl

        Save all types of poolings such that each file has a [3, emb_dim]
        where rows 0, 1, 2 are mean, max, cls pooled respectively

        Args:
            batch: Each sample contains protein_id and sequence
        """
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
        embeddings = self.get_embedding(batch)
        embeddings = embeddings.detach().cpu()

        for protein_id, emb in zip(
            batch["protein_id"], embeddings
        ):
            embeddings_dict = {
                self.pooling: emb,
            }
            with open(f"{out_dir}/{protein_id}.pkl", "wb") as file:
                pickle.dump(embeddings_dict, file)


