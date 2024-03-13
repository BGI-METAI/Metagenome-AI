#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/13/24 2:09 PM
# @Author  : zhangchao
# @File    : train_prottrans_ner_model.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import random
from framework import ProteinNERTrainer
from framework.classifier import AminoAcidsNERClassifier
from framework.prottrans import PROTTRANS_T5_TYPE
from framework.train import TRAIN_LOADER_TYPE, TEST_LOADER_TYPE

SEED = 42
TRAIN_SIZE = 0.7
BATCH_SIZE = 1
PFAM_NUM = 20794


if __name__ == '__main__':
    path = '/media/Data/zhangchao/metageomics/datasets/pfam_pkls'
    pretrained_embedding_model_name_or_path = '/media/Data/zhangchao/metageomics/weights/prot_t5_xl_half_uniref50-enc'
    files = [osp.join(path, f) for f in os.listdir(path) if f.endswith('pkl')]
    random.seed(SEED)
    random.shuffle(files)
    train_files = files[:round(len(files) * TRAIN_SIZE)]
    test_files = files[round(len(files) * TRAIN_SIZE):]

    trainer = ProteinNERTrainer(
        pretrained_embedding_model_name_or_path=pretrained_embedding_model_name_or_path,
        embedding_mode=PROTTRANS_T5_TYPE,
        do_lower_case=False,
        legacy=False
    )

    trainer.dataset_register(
        data_files=train_files,
        batch_size=BATCH_SIZE,
        mode=TRAIN_LOADER_TYPE,
    )
    trainer.dataset_register(
        data_files=test_files,
        batch_size=BATCH_SIZE,
        mode=TEST_LOADER_TYPE,
    )

    classifier = AminoAcidsNERClassifier(
        input_dims=trainer.embedding_model.get_embedding_dim,
        hidden_dims=1024,
        num_classes=PFAM_NUM
    )

    trainer.model_register(model=classifier)
    trainer.train(output_home='./output/protein_anno_NER')
