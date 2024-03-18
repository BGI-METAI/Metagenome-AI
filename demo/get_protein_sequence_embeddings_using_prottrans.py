#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 2:08 PM
# @Author  : zhangchao
# @File    : get_protein_sequence_embeddings_using_prottrans.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from framework import ParseConfig
from framework.dataset import CustomNERDataset, SequentialDistributedSampler
from framework.prottrans import ProtTransEmbeddings

if __name__ == '__main__':
    config = ParseConfig.register_parameters()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    dist.init_process_group(
        backend="nccl",
        init_method='tcp://localhost:12356',
        rank=0,
        world_size=1
    )
    torch.cuda.set_device(config.local_rank)

    pairs_path = [osp.join(config.data_path, file) for file in os.listdir(config.data_path)]

    dataset = CustomNERDataset(
        processed_sequence_label_pairs_path=pairs_path,
        tokenizer_model_name_or_path=config.model_path_or_name,
        mode=config.embed_mode,
        legacy=False,
        do_lower_case=False
    )
    test_sampler = SequentialDistributedSampler(dataset=dataset, batch_size=3)

    prottrans_model = ProtTransEmbeddings(
        model_name_or_path=config.model_path_or_name,
        mode=config.embed_mode,
        local_rank=config.local_rank,
        legacy=False,
        do_lower_case=False
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        sampler=test_sampler)

    for sample in dataloader:
        input_ids, attention_mask, batch_label = sample
        protein_vec = prottrans_model.get_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pooling='all',
            convert2numpy=False
        )
        print(protein_vec.shape)
        break
