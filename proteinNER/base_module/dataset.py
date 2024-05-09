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
            tokenizer_model_name_or_path: str,
            **kwargs
    ):
        self.pairs_path = processed_sequence_label_pairs_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_name_or_path, **kwargs)

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        with open(self.pairs_path[idx], 'rb') as file:
            data = pickle.load(file)
        return {'seq': data['seq'], 'label': data['label']}

    def collate_fn(self, batch_sample):
        batch_seq, batch_label = [], []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
            batch_label.append(sample['label'])
        batch_seq = self.prepare_sequence(batch_seq)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest'
        )

        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        for idx, val in enumerate(batch_label):
            tag_tensor = torch.tensor(val)
            batch_label[idx] = torch.nn.functional.pad(tag_tensor, (0, input_ids.shape[1] - tag_tensor.shape[0]))
        batch_label = torch.stack(batch_label)
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


class CustomPEFTEmbeddingDataset(CustomNERDataset):
    def __init__(
            self,
            incremental_protein_sequence_path: List[str],
            tokenizer_model_name_or_path: str,
            **kwargs
    ):
        super(CustomPEFTEmbeddingDataset, self).__init__(
            processed_sequence_label_pairs_path=incremental_protein_sequence_path,
            tokenizer_model_name_or_path=tokenizer_model_name_or_path,
            **kwargs
        )

    def __getitem__(self, idx):
        with open(self.pairs_path[idx], 'rb') as file:
            data = pickle.load(file)
        return {'seq': data['seq']}

    def collate_fn(self, batch_sample):
        batch_seq = []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])

        batch_seq = self.prepare_sequence(batch_seq)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest'
        )
        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        return input_ids, attention_mask


class CustomMaskDataset(Dataset):
    def __init__(
            self,
            processed_sequence_label_pairs_path: List[str],
            tokenizer_model_name_or_path: str,
            max_token: int = 512,
            **kwargs
    ):
        self.max_token = max(max_token - 1, 0)
        self.pairs_path = processed_sequence_label_pairs_path
        self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_model_name_or_path, **kwargs)
        # add mask token in tokenizer, vocab_size = 28 + 1
        self.tokenizer.add_special_tokens({"mask_token": "<extra_id_99>"})

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx):
        with open(self.pairs_path[idx], 'rb') as file:
            data = pickle.load(file)
        return {'seq': data['seq'][:self.max_token]}

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability: float = 0.15):
        """Prepare masked token input/labels for MLM: 80% MASK, 10% random, 10% original
        https://blog.csdn.net/m0_37531129/article/details/103059207"""
        labels = inputs.clone()
        masked_indices = torch.bernoulli(torch.full(labels.shape, mlm_probability)).bool()

        # only compute loss on masked tokens
        labels[~masked_indices] = 0
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        return inputs, labels

    def collate_fn(self, batch_sample):
        batch_seq = []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
        batch_seq = self.prepare_sequence(batch_seq)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest'
        )

        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.tensor(tokens['attention_mask'])
        mlm_input_ids, mlm_labels = self.mask_tokens(input_ids)
        mlm_labels *= attention_mask
        mlm_input_ids *= attention_mask

        return mlm_input_ids, attention_mask, mlm_labels

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


class NSPData(Dataset):
    def __init__(self, dataset, tokenizer=None, indices=False):
        """ Constructor
        url: https://github.com/Eryk96/NetSurfP-3.0/tree/main
        Args:
            X (np.array): The array that contains the training data
            y (np.array): The array that contains the test data
        """
        self.AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
        self.data = torch.tensor(dataset['data'][:, :, :50]).float()
        self.targets = torch.tensor(dataset['data'][:, :, 50:68]).float()
        self.lengths = torch.tensor([sum(target[:, 0] == 1) for target in self.targets])
        self.tokenizer = tokenizer

        self.unknown_nucleotide_mask()

    def unknown_nucleotide_mask(self):
        """ Augments the target with a unknown nucleotide mask
            by finding entries that don't have any residue
        """

        # creates a mask based on the one hot encoding
        unknown_nucleotides = torch.max(self.data[:, :, :20], dim=2)
        unknown_nucleotides = unknown_nucleotides[0].unsqueeze(2)

        # merge the mask to first position of the targets
        self.targets = torch.cat([self.targets, unknown_nucleotides], dim=2)

    def __getitem__(self, index):
        """ Returns train and test data at an index
        Args:
            index (int): Index at the array
        """
        X = self.data[index]
        y = self.targets[index]
        lengths = self.lengths[index]

        return X, y, lengths

    def __len__(self):
        """Returns the length of the data"""
        return len(self.data)

    def collate_fn(self, batch_sample):
        if self.tokenizer is None:
            raise NotImplementedError(f"not provide tokenizer. {self.tokenizer}")

        batch_seq, batch_label, attention_mask = [], [], []
        for X, y, lengths in batch_sample:
            seq_idx = X[:, :20].argmax(dim=1)
            seq = " ".join([self.AMINO_ACIDS[idx] for idx in seq_idx[:-1]])
            # seq_len = int(y[:, 0].sum())
            # seq = seq[:seq_len]
            batch_seq.append(seq)
            batch_label.append(y)
            attention_mask.append(y[:, 0].int())

        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=batch_seq,
            padding='longest',
            max_length=y.shape[0]
        )

        input_ids = torch.tensor(tokens['input_ids'])
        attention_mask = torch.stack(attention_mask)
        # for idx, val in enumerate(batch_label):
        #     tag_tensor = torch.tensor(val)
        #     batch_label[idx] = torch.nn.functional.pad(tag_tensor, (0, input_ids.shape[1] - tag_tensor.shape[0]))
        batch_label = torch.stack(batch_label)
        return input_ids, attention_mask, batch_label
