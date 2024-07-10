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
import torch
from torch.nn import Identity
from huggingface_hub import login
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
from huggingface_hub.hf_api import HfFolder

from embedding_esm import EsmEmbedding

class Esm3Embedding(EsmEmbedding):
    def __init__(self, pooling="mean"):
        login(token="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")
        # HfFolder.save_token(access_token)

        # This will download the model weights and instantiate the model on your machine.
        model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda") # or "cpu"

        self.model = model
        self.alphabet = EsmSequenceTokenizer()
        self.model.output_heads = Identity()
        self.model.structure_decoder = Identity()
        self.model.function_decoder = Identity()
        self.model.structure_encoder = Identity()
        self.model.eval()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.embed_dim = model.embed_dim
        self.pooling = pooling
        # Freeze the parameters of ESM
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    