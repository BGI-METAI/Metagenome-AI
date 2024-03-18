#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/13/24 2:09 PM
# @Author  : zhangchao
# @File    : train_prottrans_ner_model.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import random
import torch
import torch.multiprocessing as mp
from framework import ProteinNERTrainer, ParseConfig
from framework.classifier import AminoAcidsNERClassifier
from framework.base_train import TRAIN_LOADER_TYPE, TEST_LOADER_TYPE
from framework.prottrans import ProtTransEmbeddings


def test(world_size):
    config = ParseConfig.register_parameters()

    files = [osp.join(config.data_path, f) for f in os.listdir(config.data_path) if f.endswith('pkl')]
    random.seed(config.seed)
    random.shuffle(files)
    train_files = files[:round(len(files) * config.train_size)]
    test_files = files[round(len(files) * config.train_size):]

    trainer = ProteinNERTrainer()
    trainer.ddp_register(rank=config.rank, world_size=world_size)

    trainer.dataset_register(
        data_files=train_files,
        batch_size=config.batch_size,
        mode=TRAIN_LOADER_TYPE,
        tokenizer_model_name_or_path=config.model_path_or_name,
        tokenizer_mode=config.embed_mode,
        legacy=False,
        do_lower_case=False
    )
    trainer.dataset_register(
        data_files=test_files,
        batch_size=config.batch_size,
        mode=TEST_LOADER_TYPE,
        tokenizer_model_name_or_path=config.model_path_or_name,
        tokenizer_mode=config.embed_mode,
        legacy=False,
        do_lower_case=False
    )

    embedding_model = ProtTransEmbeddings(
        model_name_or_path=config.model_path_or_name,
        mode=config.embed_mode,
        local_rank=config.local_rank
    )

    classifier = AminoAcidsNERClassifier(
        input_dims=trainer.embedding_model.get_embedding_dim,
        hidden_dims=1024,
        num_classes=config.num_classes
    )

    trainer.model_register(model=embedding_model, local_rank=config.local_rank)
    trainer.model_register(model=classifier, local_rank=config.local_rank)
    trainer.train(
        output_home=config.output_home
    )


if __name__ == '__main__':
    world_size=torch.cuda.device_count()
    mp.spawn(test, args=(world_size,), nprocs=world_size)
