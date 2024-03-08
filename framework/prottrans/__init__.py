#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/7/24 5:49 PM
# @Author  : zhangchao
# @File    : __init__.py.py
# @Email   : zhangchao5@genomics.cn

PROTTRANS_T5_TYPE: str = 't5'
PROTTRANS_BERT_TYPE: str = 'bert'
PROTTRANS_ALBERT_TYPE: str = 'albert'
PROTTRANS_XLENT_TYPE: str = 'xlent'

POOLING_CLS_TYPE: str = 'cls'
POOLING_MEAN_TYPE: str = 'mean'
POOLING_SUM_TYPE: str = 'sum'

from .prottrans_embeddings import ProtTransEmbeddings
