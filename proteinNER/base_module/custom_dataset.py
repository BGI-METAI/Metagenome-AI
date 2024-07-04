#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : custom_dataset.py
# @Author  : zhangchao
# @Date    : 2024/7/4 14:00 
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import pickle
import re
import random
import torch
from typing import List

from torch.utils.data import Dataset
from transformers import T5Tokenizer
from tokenizers import Tokenizer

from proteinNER.base_module.custom_tokenizer import CustomTokenizer


class CustomDiffusionDataset(Dataset):
    def __init__(
            self,
            protein_seq_and_label_path_list,
            sequence_model_name_or_path,
            label_name_or_path,
            **kwargs
    ):
        self.data_paths = protein_seq_and_label_path_list
        self.label_masked_rate = kwargs.get('label_masked_rate', 0.1)
        self.sequence_tokenizer = T5Tokenizer.from_pretrained(sequence_model_name_or_path)
        self.label_tokenizer = CustomTokenizer.from_pretrained(label_name_or_path)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data = pickle.load(open(self.data_paths[idx], 'rb'))
        raw_label, masked_label = self.prepare_label_sequence(data['token_label'])
        sequence = self.prepare_protein_sequence(data['seq'])
        return {'seq': sequence, 'raw_label': raw_label, 'masked_label': masked_label}

    def collate_fn(self, batch_sample):
        batch_seq, batch_raw_label, batch_masked_label = [], [], []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
            batch_raw_label.append(sample['raw_label'])
            batch_masked_label.append(sample['masked_label'])
        sequence_tokens = self.sequence_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest',
            add_special_tokens=False
        )
        raw_label_tokens = self.label_tokenizer.batch_encode_plus(batch_raw_label, add_special_tokens=False)
        masked_label_tokens = self.label_tokenizer.batch_encode_plus(batch_masked_label, add_special_tokens=False)
        return (
            {
                'input_ids': torch.tensor(sequence_tokens['input_ids']),
                'attention_mask': torch.tensor(sequence_tokens['attention_mask'])
            }, {
                'input_ids': torch.tensor(masked_label_tokens.ids),
                'attention_mask': torch.tensor(masked_label_tokens.attention_mask)
            }, {
                'input_ids': torch.tensor(raw_label_tokens.ids),
                'attention_mask': torch.tensor(raw_label_tokens.attention_mask)
            })

    @staticmethod
    def prepare_protein_sequence(sequence):
        return ' '.join(re.sub(r'[UZOB]', 'X', sequence))

    def prepare_label_sequence(self, labels: List):
        labels = [x.split('-')[1] if x != 'O' else x for x in labels]
        raw_label = labels.copy()
        length = len(labels)
        indices = list(range(length))
        random.shuffle(indices)
        mask_ids = indices[:int(round(length * self.label_masked_rate))]
        for idx in mask_ids:
            labels[idx] = '[MASK]'
        return raw_label, labels
