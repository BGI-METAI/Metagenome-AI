#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 2:59 PM
# @Author  : zhangchao
# @File    : get_protrain_sequence_embeddings_using_esm2.py
# @Email   : zhangchao5@genomics.cn
import pandas as pd

from framework.esm2 import Esm2Embeddings

if __name__ == '__main__':
    demo_sequences = [
        ("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
        ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein2 with mask", "KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
        ("protein3", "K A <mask> I S Q"),
    ]

    esm2_model = Esm2Embeddings()

    protein_embed = esm2_model.get_embedding(batch=demo_sequences)

    print(protein_embed.shape)
