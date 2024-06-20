#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:56 PM
# @Author  : zhangchao
# @File    : dataset.py
# @Email   : zhangchao5@genomics.cn
import re
import os.path as osp
import pickle
import torch

from typing import Optional, Union, List
from torch.utils.data import Dataset
from transformers import T5Tokenizer


class CustomNERDataset(Dataset):
    def __init__(
            self,
            processed_sequence_label_pairs_path: List[str],
            label2id_path: str,
            tokenizer_model_name_or_path: str,
            **kwargs
    ):
        self.pairs_path = processed_sequence_label_pairs_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_name_or_path, **kwargs)
        self.label2id = pickle.load(open(label2id_path, 'rb'))

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        with open(self.pairs_path[idx], 'rb') as file:
            data = pickle.load(file)
        label = []
        for tag in data['token_label']:
            label.append(self.label2id[tag])
        return {'seq': data['seq'], 'label': torch.tensor(label),
                'protein_id': osp.splitext(osp.basename(self.pairs_path[idx]))[0]}

    def collate_fn(self, batch_sample, is_valid):
        batch_seq, batch_label, batch_protein_id = [], [], []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
            batch_label.append(sample['label'])
            batch_protein_id.append(sample['protein_id'])
        batch_seq = self.prepare_sequence(batch_seq)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest',
            add_special_tokens=False
        )
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])

        if not is_valid:
            batch_label = torch.nn.utils.rnn.pad_sequence(batch_label, batch_first=True, padding_value=0)
            batch_label = torch.tensor(batch_label)
            return input_ids, attention_mask, batch_label
        else:
            return input_ids, attention_mask, batch_label, batch_protein_id

    @staticmethod
    def prepare_sequence(
            sequence: Optional[Union[str, List[str]]],
            add_separator: bool = True
    ) -> List[str]:
        """
        prepare protein sequence, add a space separator between each amino acid character and replace rare amino acids with `X`.

        :param sequence:
            amino acid sequences
        :param add_separator:
            whether to add space delimiters to each sequence. default is True.
        :return:
        """
        if isinstance(sequence, List):
            sequence = [re.sub(r'[UZOB]', 'X', seq) for seq in sequence]

        elif isinstance(sequence, str):
            sequence = [re.sub(r'[UZOB]', 'X', sequence)]
        else:
            raise ValueError('Error: Got an invalid protein sequence!')

        if add_separator:
            return [' '.join(list(seq)) for seq in sequence]
        else:
            return sequence


class InferDataset(CustomNERDataset):
    def __init__(
            self,
            processed_sequence_label_pairs_path: List[str],
            tokenizer_model_name_or_path: str,
            **kwargs
    ):
        self.pairs_path = processed_sequence_label_pairs_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_name_or_path, **kwargs)

    def __getitem__(self, idx):
        with open(self.pairs_path[idx], 'rb') as file:
            data = pickle.load(file)
        return {'seq': data['seq'], 'protein_id': osp.splitext(osp.basename(self.pairs_path[idx]))[0]}

    def collate_fn(self, batch_sample, **kwargs):
        batch_seq, batch_protein_id = [], []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
            batch_protein_id.append(sample['protein_id'])
        batch_seq = self.prepare_sequence(batch_seq)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest',
            add_special_tokens=False
        )
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])

        return input_ids, attention_mask, batch_protein_id


class DiscriminatorDataset(Dataset):
    def __init__(self, data_list, max_length=250, **kwargs):
        self.data_list = data_list
        self.max_length = max_length

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        with open(self.data_list[idx], 'rb') as fp:
            record = pickle.load(fp)
            data = record['crf_label']
            data.extend([0] * (self.max_length - len(data)))
        label = [0 if 'cross' in self.data_list[idx] else 1]
        return {'data': data, 'label': label}

    def collate_fn(self, batch_sample):
        batch_data, batch_label = [], []
        for sample in batch_sample:
            batch_data.append(sample['data'])
            batch_label.append(sample['label'])

        batch_data = torch.tensor(batch_data, dtype=torch.float)
        batch_label = torch.tensor(batch_label).flatten()

        return batch_data, batch_label
