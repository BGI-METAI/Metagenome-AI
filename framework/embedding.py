#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   embedding.py
@Time    :   2024/03/06 21:02:02
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

from abc import ABC, abstractmethod


class Embedding(ABC):
    @abstractmethod
    def get_embedding(self):
        raise NotImplementedError()

    @abstractmethod
    def get_embedding_dim(self, batch=None):
        raise NotImplementedError()

    @abstractmethod
    def to(self):
        raise NotImplementedError()
