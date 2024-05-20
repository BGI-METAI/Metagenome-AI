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

import csv
import torch
from torch.utils.data import Dataset
from collections import defaultdict


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


class TSVDataset(Dataset):
    def __init__(self, path):
        with open(path) as file:
            reader = csv.reader(file, delimiter=" ", quotechar='"')
            self.samples = list(reader)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {"protein_id": sample[0], "len": sample[1], "sequence": sample[2]}


class MaxTokensLoader:
    def __init__(self, dataset, max_tokens, start_ind, chunk_size, drop_last=False):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.start_ind = start_ind
        self.chunk_size = chunk_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = defaultdict(list)
        total_tokens = 0
        for idx in range(self.start_ind, self.start_ind + self.chunk_size):
            sample = self.dataset[idx]
            batch["protein_id"].append(sample["protein_id"])
            batch["sequence"].append(sample["sequence"])
            total_tokens += int(sample["len"])
            if total_tokens > self.max_tokens:
                yield batch
                batch = defaultdict(list)
                total_tokens = 0
        if batch.size(0) > 0 and not self.drop_last:
            yield batch
