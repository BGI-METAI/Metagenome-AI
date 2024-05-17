#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:56 PM
# @Author  : zhangchao
# @File    : dataset.py
# @Email   : zhangchao5@genomics.cn
import re
import math
import pickle
import torch
import torch.distributed as dist

from typing import Optional, Union, List
from torch.utils.data import Dataset, Sampler
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
        return {'seq': data['seq'], 'label': torch.tensor(label)}

    def collate_fn(self, batch_sample):
        batch_seq, batch_label = [], []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
            batch_label.append(sample['label'])
        batch_seq = self.prepare_sequence(batch_seq)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest',
            add_special_tokens=False
        )
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        batch_label = torch.nn.utils.rnn.pad_sequence(batch_label, batch_first=True, padding_value=0)
        batch_label = torch.tensor(batch_label)
        return input_ids, attention_mask, batch_label

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


class SequentialDistributedSampler(Sampler):
    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        super(SequentialDistributedSampler, self).__init__(data_source=dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples: (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples

