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
import pickle
from sklearn.preprocessing import MultiLabelBinarizer


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
    def __init__(self, path, embeddings_dir=None, emb_type=None):
        if embeddings_dir is not None and emb_type is None:
            raise ValueError("You must provide emb_type as well")
        self.embeddings_dir = embeddings_dir
        self.emb_type = emb_type
        with open(path) as file:
            reader = csv.reader(file, delimiter=" ")
            self.samples = list(reader)
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([sample[3:] for sample in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prot_id = sample[0]
        if self.embeddings_dir is not None:
            with open(f"{self.embeddings_dir}/{prot_id}.pkl", "rb") as file_emb:
                prot_emb = pickle.load(file_emb)
        return {
            "protein_id": sample[0],
            "len": sample[1],
            "sequence": sample[2],
            "emb": (
                torch.from_numpy(prot_emb[self.emb_type])
                if self.embeddings_dir is not None
                else None
            ),
            "labels": torch.from_numpy(self.mlb.transform([sample[3:]])).to(
                torch.float32
            ),
        }

    def get_number_of_labels(self):
        return len(self.mlb.classes_)


class MaxTokensLoader:
    def __init__(self, dataset, max_tokens, start_ind, end_ind, drop_last=False):
        self.dataset = dataset
        self.max_tokens = max_tokens
        self.start_ind = start_ind
        self.end_ind = end_ind
        self.drop_last = drop_last

    def __iter__(self):
        batch = defaultdict(list)
        total_tokens = 0
        for idx in range(self.start_ind, self.end_ind):
            try:
                sample = self.dataset[idx]
                batch["protein_id"].append(sample["protein_id"])
                batch["sequence"].append(sample["sequence"])
                total_tokens += int(sample["len"])
                if total_tokens > self.max_tokens:
                    yield batch
                    batch = defaultdict(list)
                    total_tokens = 0
            except IndexError:
                print(f"Attempted to access index {idx} of dataset.")
        if len(batch) > 0 and not self.drop_last:
            yield batch


class LoadStoredDataset(Dataset):
    def __init__(self, path, embeddings_dir, emb_type):
        with open(path) as file:
            reader = csv.reader(file, delimiter=" ", quotechar='"')
            self.samples = list(reader)
        self.embeddings_dir = embeddings_dir
        self.emb_type = emb_type

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        prot_id = sample[0]
        with open(f"{self.embeddings_dir}/{prot_id}.pkl", "rb") as file_emb:
            prot_emb = pickle.load(file_emb)
        return prot_emb[self.emb_type]
