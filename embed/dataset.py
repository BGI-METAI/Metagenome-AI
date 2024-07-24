import csv
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
import os
from .embedding_esm3 import Esm3Embedding
import pandas as pd


class TSVDataset(Dataset):
    def __init__(self, path, embeddings_dir=None, emb_type=None):
        if embeddings_dir is not None and emb_type is None:
            raise ValueError("You must provide emb_type as well")
        self.embeddings_dir = embeddings_dir
        self.emb_type = emb_type
        with open(path) as file:
            reader = csv.reader(file, delimiter=" ")
            self.header = list(reader)[0]
            self.data = list(reader)[1:]

        # 适应 多机多卡
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([sample[3:] for sample in self.data])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        protein_id = sample[0]
        label = torch.tensor(sample[2])

        return [sample[0], sample[1], sample[2]]
    
    def get_number_of_labels(self):
        return len(self.mlb.classes_)

class CSVDataset(Dataset):
    def __init__(self, path):

        self.df = pd.read_csv(path, sep=',')
        # 确保  'Sequence' 列存在
        assert 'Sequence' in self.df.columns, "CSV file must contain 'Sequence' column"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 获取指定索引的行
        row = self.df.iloc[idx]

        # 获取蛋白质序列，并去除换行符
        sequence = row['Sequence'].replace("\n", "")

        # 获取蛋白质 ID 和名称
        protein_id = row['ID']
        lable = row['lable']
        # emb = Esm3Embedding(sequence)
        # label = 1 为Amp , 0为Non-Amp
        # lable = 0
        return {'sequence': sequence, 'protein_id': protein_id,  "label": lable, "len": len(sequence)}

    def get_number_of_labels(self):
        return len(self.df['lable'].value_counts())


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