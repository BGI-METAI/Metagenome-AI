#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/27 13:50
# @Author  : zhangchao
# @File    : demo_peft_ner_t5.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import sys
import random
from warnings import filterwarnings

from framework.finetuning.finetuning_prottrans import FineTuneProtTransAAModel
from framework.finetuning.finetuning_train import PEFTProteinT5Trainer
from framework.finetuning.parse_configuration import ParsePEFTConfig

# sys.path.insert(0, '..')
filterwarnings('ignore')


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # register hyper-parameters
    config = ParsePEFTConfig.register_parameters()

    # prepare dataset
    files = [osp.join(config.data_path, f) for f in os.listdir(config.data_path) if f.endswith('pkl')]
    random.seed(config.seed)
    random.shuffle(files)
    train_files = files[:round(len(files) * config.train_size)]
    test_files = files[round(len(files) * config.train_size):]

    # initialize trainer class
    trainer = PEFTProteinT5Trainer(config)

    # register dataset
    trainer.dataset_register(
        data_files=train_files,
        batch_size=config.batch_size,
        data_type='train',
        tokenizer_model_name_or_path=config.model_path_or_name,
        legacy=False,
        do_lower_case=False
    )

    trainer.dataset_register(
        data_files=test_files,
        batch_size=config.batch_size,
        data_type='test',
        tokenizer_model_name_or_path=config.model_path_or_name,
        legacy=False,
        do_lower_case=False
    )

    # instantiate and register model
    model = FineTuneProtTransAAModel(
        model_name_or_path=config.model_path_or_name,
        n_classes=config.num_classes
    )
    trainer.model_register(model=model, model_type='cls')

    trainer.train(config)
