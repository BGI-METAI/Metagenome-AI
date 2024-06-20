#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : demo_discriminator_train.py
# @Author  : zhangchao
# @Date    : 2024/6/5 18:07 
# @Email   : zhangchao5@genomics.cn
import argparse
import random
import sys

sys.path.insert(0, "..")
sys.path.insert(0, "/home/share/huadjyin/home/zhangkexin2/model/")

from proteinNER.classifier.model import ProteinDiscriminator
from proteinNER.classifier.trainer import DiscriminatorTrainer


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Predicted Results Discriminator')
    parser.add_argument('--train_data_path', type=str, required=True, help='the path of input train dataset')
    parser.add_argument('--test_data_path', type=str, required=True, help='the path of input test dataset')
    parser.add_argument('--output_home', type=str, required=True, help='the output data home')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    parser.add_argument('--batch_size', type=int, default=100, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--k', type=int, default=100, help='gradient accumulation parameters')
    parser.add_argument('--lr_decay_step', type=int, default=2, help='Period of learning rate decay')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.99,
                        help='multiplicative factor of learning rate decay')

    parser.add_argument('--mode', default='best',
                        help='whether to load the optimal model, effective when reuse is true')
    parser.add_argument('--max_length', type=int, default=300, help='the maximum length of input data')
    parser.add_argument('--is_trainable', action='store_true', help='whether the model should be trainable or not')
    parser.add_argument('--reuse', default=False)

    parser.add_argument('--user_name', type=str, default='kxzhang2000', help='wandb register parameter')
    parser.add_argument('--project', type=str, default='ProteinDiscriminator', help='wandb project name')
    parser.add_argument('--group', type=str, default='accelerate', help='wandb group')

    return parser.parse_args()


def worker():
    args = register_parameters()

    train_data_files = []
    with open(args.train_data_path, 'r') as fp:
        for line in fp.readlines():
            train_data_files.append(line.strip())
    random.seed(args.seed)
    random.shuffle(train_data_files)

    test_data_files = []
    with open(args.test_data_path, 'r') as fp:
        for line in fp.readlines():
            test_data_files.append(line.strip())
    random.seed(args.seed)
    random.shuffle(test_data_files)

    trainer = DiscriminatorTrainer(output_home=args.output_home, k=args.k)
    trainer.register_dataset(data_files=train_data_files, batch_size=args.batch_size, max_length=args.max_length, mode='train')
    trainer.register_dataset(data_files=test_data_files, batch_size=args.batch_size, max_length=args.max_length,mode='test')

    model = ProteinDiscriminator(input_dims=args.max_length)
    trainer.register_model(
        model=model,
        reuse=args.reuse,
        is_trainable=args.is_trainable,
        learning_rate=args.learning_rate,
        mode=args.mode,
        lr_decay_step=args.lr_decay_step,
        lr_decay_gamma=args.lr_decay_gamma
    )
    if args.is_trainable:
        trainer.print_trainable_parameters()
        trainer.train(**vars(args))
    else:
        trainer.valid_model_performance()


if __name__ == '__main__':
    worker()