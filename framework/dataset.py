#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:56 PM
# @Author  : zhangchao
# @File    : dataset.py
# @Email   : zhangchao5@genomics.cn
import pandas as pd

from torch.utils.data import Dataset


class CustomDataFrameDataset(Dataset):
    def __init__(
            self,
            dataset: pd.DataFrame,
            sequence_key: str = 'sequence',
            target_key: str = 'label'
    ):
        self.dataset = dataset
        self.sequence_list = self.dataset[sequence_key].tolist()
        self.target_list = self.dataset[target_key].tolist()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.sequence_list[idx], self.target_list[idx]
