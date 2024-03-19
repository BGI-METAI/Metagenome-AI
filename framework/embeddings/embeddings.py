#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/7/24 4:58 PM
# @Author  : zhangchao
# @File    : embeddings.py
# @Email   : zhangchao5@genomics.cn
import numpy as np
from abc import ABC, abstractmethod


class Embeddings(ABC):
    """Interface for protein language models (pLMs)"""

    @abstractmethod
    def get_embedding(self, **kwargs) -> np.ndarray:
        """Obtain the protein sequence embeddings."""

    @abstractmethod
    def get_embedding_dim(self, **kwargs) -> int:
        """Obtain the protein sequence embedding dimension of pLMs."""

    @abstractmethod
    def to(self, device):
        """transfer model to device"""
