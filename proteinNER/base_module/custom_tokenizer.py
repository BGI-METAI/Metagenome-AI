#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : custom_tokenizer.py
# @Author  : zhangchao
# @Date    : 2024/7/4 15:18 
# @Email   : zhangchao5@genomics.cn
import os.path as osp

from tokenizers import Tokenizer


class CustomBatchTokenizerEncoding:
    def __init__(self, custom_tokens):
        self.custom_tokens = custom_tokens

    @property
    def ids(self):
        return [x.ids for x in self.custom_tokens]

    @property
    def type_ids(self):
        return [x.type_ids for x in self.custom_tokens]

    @property
    def tokens(self):
        return [x.tokens for x in self.custom_tokens]

    @property
    def offsets(self):
        return [x.offsets for x in self.custom_tokens]

    @property
    def attention_mask(self):
        return [x.attention_mask for x in self.custom_tokens]

    @property
    def special_tokens_mask(self):
        return [x.special_tokens_mask for x in self.custom_tokens]

    @property
    def overflowing(self):
        return [x.overflowing for x in self.custom_tokens]


class CustomTokenizer:
    def __init__(self, pretrained_model_name_or_path):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.tokenizer = Tokenizer.from_file(osp.join(self.pretrained_model_name_or_path, 'tokenizer.json'))

    def __call__(self):
        return self.tokenizer

    def encode(self, sequence, add_special_tokens=False):
        return self.tokenizer.encode(sequence, add_special_tokens=add_special_tokens)

    def decode(self, ids, skip_special_tokens=True):
        self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def batch_encode_plus(self, batch_sequence, padding='longest', add_special_tokens=False):
        if padding == 'longest':
            padding_length = max(len(seq) for seq in batch_sequence)
        else:
            raise ValueError('Can not support current padding mode!')
        pad_sequence = [x + ['[PAD]'] * (padding_length - len(x)) for x in batch_sequence]
        tokens = [self.encode(' '.join(x), add_special_tokens) for x in pad_sequence]
        return CustomBatchTokenizerEncoding(custom_tokens=tokens)

    def batch_decode_plus(self, batch_ids, skip_special_tokens=True):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):
        tokenizer = cls(pretrained_model_name_or_path=pretrained_model_name_or_path)
        return tokenizer
