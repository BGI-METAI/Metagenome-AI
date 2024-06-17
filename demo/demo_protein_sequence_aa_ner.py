#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:49
# @Author  : zhangchao
# @File    : demo_protein_sequence_aa_ner.py
# @Email   : zhangchao5@genomics.cn
import argparse
import sys
import random

sys.path.insert(0, "..")
sys.path.insert(0, "/home/share/huadjyin/home/zhangchao5/code/ProtT5")

from proteinNER.classifier.model import ProtT5Conv1dCRF4AAClassifier
from proteinNER.classifier.trainer import ProteinAANERTrainer


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument('--train_data_path', type=str, required=True, help='the path of input dataset')
    parser.add_argument('--label2id_path', type=str, required=True, help='the path of label to id path')
    parser.add_argument('--test_data_path', type=str, required=True, help='the path of input dataset')
    parser.add_argument('--output_home', type=str, required=True, help='the path of output')
    parser.add_argument('--model_path_or_name', type=str, required=True, help='pretrianed pLM model path or name')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--num_classes', type=int, required=True, help='the number of categories')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--loss_weight', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--k', type=int, default=100, help='Gradient accumulation parameters')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Period of learning rate decay')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.99, help='Multiplicative factor of learning rate decay')
    parser.add_argument('--reuse', action='store_true')
    parser.add_argument('--mode', default='best',
                        help='whether to load the optimal model, effective when reuse is true')
    parser.add_argument('--is_trainable', action='store_true',
                        help='Whether the LoRA adapter should be trainable or not.')
    parser.add_argument('--is_valid', action='store_true',
                        help='Whether to verify model performance')

    parser.add_argument('--user_name', type=str, default='kxzhang2000', help='wandb register parameter')
    parser.add_argument('--project', type=str, default='ProteinConvCRF', help='wandb project name')
    parser.add_argument('--group', type=str, default='valid', help='wandb group')

    return parser.parse_args()


def worker():
    args = register_parameters()

    # prepare dataset
    train_files = []
    with open(args.train_data_path, 'r') as file:
        for line in file.readlines():
            train_files.append(line.strip())
    random.seed(args.seed)
    random.shuffle(train_files)

    test_files = []
    with open(args.test_data_path, 'r') as file:
        for line in file.readlines():
            test_files.append(line.strip())

    # initialize trainer class
    trainer = ProteinAANERTrainer(output_home=args.output_home, k=args.k)

    # register dataset
    trainer.register_dataset(
        data_files=train_files,
        label2id_path=args.label2id_path,
        mode='train',
        dataset_type='class',
        batch_size=args.batch_size,
        model_name_or_path=args.model_path_or_name
    )

    trainer.register_dataset(
        data_files=test_files[:10000],
        label2id_path=args.label2id_path,
        mode='test',
        dataset_type='class',
        batch_size=args.batch_size,
        is_valid=args.is_valid,
        model_name_or_path=args.model_path_or_name
    )

    model = ProtT5Conv1dCRF4AAClassifier(model_name_or_path=args.model_path_or_name, num_classes=args.num_classes)
    trainer.register_model(
        model=model,
        reuse=args.reuse,
        is_trainable=args.is_trainable,
        learning_rate=args.learning_rate,
        mode=args.mode,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma
    )

    if not args.is_valid:
        trainer.print_trainable_parameters()
        trainer.train(**vars(args))
    else:
        trainer.valid_model_performance()


if __name__ == '__main__':
    worker()
