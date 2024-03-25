#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/13/24 2:09 PM
# @Author  : zhangchao
# @File    : demo_ner_train.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import sys
import random
from warnings import filterwarnings

sys.path.insert(0, '..')

from framework import ProteinNERTrainer, ParseConfig
from framework.classifier import AminoAcidsNERClassifier
from framework.base_train import TRAIN_LOADER_TYPE, TEST_LOADER_TYPE
from framework.prottrans import ProtTransEmbeddings

filterwarnings('ignore')

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    # loading hyper-parameters
    config = ParseConfig.register_parameters()

    # prepare dataset
    files = [osp.join(config.data_path, f) for f in os.listdir(config.data_path) if f.endswith('pkl')]
    random.seed(config.seed)
    random.shuffle(files)
    train_files = files[:round(len(files) * config.train_size)]
    test_files = files[round(len(files) * config.train_size):]

    # initialize trainer class
    trainer = ProteinNERTrainer(config)

    # register dataset
    trainer.dataset_register(
        data_files=train_files,
        batch_size=config.batch_size,
        data_type=TRAIN_LOADER_TYPE,
        tokenizer_model_name_or_path=config.model_path_or_name,
        tokenizer_mode=config.embed_mode,
        legacy=False,
        do_lower_case=False
    )

    trainer.dataset_register(
        data_files=test_files,
        batch_size=config.batch_size,
        data_type=TEST_LOADER_TYPE,
        tokenizer_model_name_or_path=config.model_path_or_name,
        tokenizer_mode=config.embed_mode,
        legacy=False,
        do_lower_case=False
    )

    # instantiate and register embedded model
    embedding_model = ProtTransEmbeddings(
        model_name_or_path=config.model_path_or_name,
        mode_type=config.embed_mode,
    )
    trainer.model_register(
        model=embedding_model,
        model_type='emb',
    )

    # instantiate and register classifier model
    classifier_model = AminoAcidsNERClassifier(
        input_dims=trainer.embedding_model.get_embedding_dim,
        num_classes=config.num_classes
    )
    trainer.model_register(
        model=classifier_model,
        model_type='cls',
    )

    trainer.train(config)
