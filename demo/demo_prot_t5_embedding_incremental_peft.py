#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/10/24 2:27 PM
# @Author  : zhangchao
# @File    : demo_prot_t5_embedding_incremental_peft.py
# @Email   : zhangchao5@genomics.cn
import argparse

from proteinNER.Incremental_finetuning.trainer import ProtT5EmbeddingIncrementalTrainer
from proteinNER.classifier.model import ProtTransT5EmbeddingPEFTModel


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='/media/Data/zhangchao/metageomics/datasets/gene3d_short_sub10w',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/media/Data/zhangchao/metageomics/datasets/gene3d_short_sub10w',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--output_home',
        type=str,
        default='/media/Data/zhangchao/metageomics/codes/Metagenome-AI/output',
    )
    parser.add_argument(
        '--model_path_or_name',
        type=str,
        default='/media/Data/zhangchao/metageomics/weights/prot_t5_xl_half_uniref50-enc',
        help='pretrianed pLM model path or name'
    )

    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_classes', type=int, default=6595,
                        help='the number of categories')  # PFAM: 20794, GENE3D: 6595
    parser.add_argument('--add_background', type=bool, default=True, help='add background type to the final categories')

    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--loss_weight', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--load_best_model', type=bool, default=True)
    parser.add_argument('--reuse', type=bool, default=False)
    parser.add_argument('--is_trainable', type=bool, default=True,
                        help='Whether the LoRA adapter should be trainable or not.')

    parser.add_argument('--user_name', type=str, default='zhangchao162', help='wandb register parameter')
    parser.add_argument('--project', type=str, default='ProteinT5Incremental', help='wandb project name')
    parser.add_argument('--group', type=str, default='IncrementalEmbed', help='wandb group')

    return parser.parse_args()


def worker():
    args = register_parameters()

    train_files = []
    # with open(args.train_data_path, 'r') as file:
    #     for line in file.readlines():
    #         train_files.extend(line.strip().split(' '))

    test_files = []
    # with open(args.test_data_path, 'r') as file:
    #     for line in file.readlines():
    #         test_files.extend(line.strip().split(' '))

    trainer = ProtT5EmbeddingIncrementalTrainer(output_home=args.output_home)

    trainer.register_dataset(
        data_files=train_files,
        mode='train',
        batch_size=args.batch_size,
        model_name_or_path=args.model_path_or_name,
        dataset_type='embed'
    )

    model = ProtTransT5EmbeddingPEFTModel(
        model_name_or_path=args.model_path_or_name,
        lora_inference_mode=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    trainer.register_model(model=model, reuse=args.reuse, is_trainable=args.is_trainable)
    trainer.train(**vars(args))


if __name__ == '__main__':
    worker()
