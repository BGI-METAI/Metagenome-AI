#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : demo_diffusion4ProteinNER.py
# @Author  : zhangchao
# @Date    : 2024/7/2 14:13 
# @Email   : zhangchao5@genomics.cn
import argparse

from proteinNER.diffusion.diffusion_trainer import DiffusionProteinFuncTrainer


def register_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='training data files'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        required=True,
        help='learning rate'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        required=True,
        help='batch size'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        required=True,
        help='the pre-trained protein language model name or path'
    )
    parser.add_argument(
        '--label2id_path',
        type=str,
        required=True,
        help='label2id file'
    )
    parser.add_argument(
        '--num_timestep',
        type=int,
        required=True,
        help='the forward diffusion total steps'
    )
    parser.add_argument(
        '--num_label',
        type=int,
        required=True,
        help='the number of protein categories'
    )
    parser.add_argument(
        '--is_trainable',
        action='store_true',
        help='if true, the model parameters is trainable'
    )
    parser.add_argument(
        '--mode',
        default='train',
        help='data loader type, optional, only support `train` and `test`'
    )
    parser.add_argument(
        '--reuse_params',
        action='store_true',
        help='if true, reload the saved checkpoint to initialize the model parameters'
    )
    parser.add_argument(
        '--decay_gamma',
        type=float,
        default=0.99,
        help='multiplicative factor of learning rate decay'
    )
    parser.add_argument(
        '--decay_step',
        type=int,
        default=10,
        help='period of learning rate decay'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='random seed'
    )
    parser.add_argument(
        '--output_home',
        type=str,
        required=True,
        help='the model output home'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=100,
        help='period of gradient accumulation'
    )
    parser.add_argument(
        '--user_name',
        type=str,
        required=True,
        help='wandb user name'
    )
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='wandb project name'
    )
    parser.add_argument(
        '--group',
        type=str,
        required=True,
        help='wandb group name'
    )

    return parser.parse_args()


def worker():
    args = register_args()
    trainer = DiffusionProteinFuncTrainer(**vars(args))
    trainer.train()


if __name__ == '__main__':
    worker()
