#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   dataset.py
@Time    :   2024/03/06 21:02:16
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.label = config["label"]
        self.sequence = config["sequence"]
        self.max_seq_len = config["max_seq_len"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_label_pair = self.dataset[idx]
        sequence = (
            seq_label_pair[self.sequence][: self.max_seq_len]
            if self.max_seq_len
            else seq_label_pair[self.sequence]
        )
        sample = {
            "sequence": sequence,
            "target": seq_label_pair["le_" + self.label],
        }
        return sample
    
# TODO take chunk00 and create .pkl files for each via store_embeddings
