#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:49
# @Author  : zhangchao
# @File    : demo_protein_sequence_aa_ner.py
# @Email   : zhangchao5@genomics.cn
import argparse
import sys
import random
import os

sys.path.insert(0, "..")

from proteinNER.classifier.model import ProtTransT5ForAAClassifier
from proteinNER.classifier.trainer import ProteinNERTrainer


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument(
        '--train_data_path',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/dataset/gene3d/gene3d.train/chunk500w/chunk1.txt',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--test_data_path',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/dataset/gene3d/gene3d.test/chunk00.00.txt',
        help='the path of input dataset'
    )
    parser.add_argument(
        '--label_dict_path',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/dataset/GENE3D_id2label.pkl',
    )
    parser.add_argument(
        '--output_home',
        type=str,
        default='/home/share/huadjyin/home/zhangkexin2/OUTPUT',
    )
    parser.add_argument(
        '--model_path_or_name',
        type=str,
        default='/home/share/huadjyin/home/zhangchao5/weight/prot_t5_xl_half_uniref50-enc',
        help='pretrianed pLM model path or name'
    )
    parser.add_argument('--inference_length_threshold', type=int, default=5,
                        help='inference domain length threshold')  # 50
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=3, help='batch size')
    parser.add_argument('--num_classes', type=int, default=6595,
                        help='the number of categories')  # PFAM: 20794, GENE3D: 6595
    parser.add_argument('--add_background', type=bool, default=True, help='add background type to the final categories')

    parser.add_argument('--epoch', type=int, default=10)  # 100
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--loss_weight', type=float, default=1.)
    parser.add_argument('--patience', type=int, default=1)
    parser.add_argument('--k', type=int, default=200, help='Gradient accumulation parameters')
    parser.add_argument('--load_best_model', type=bool, default=True)
    parser.add_argument('--reuse', type=bool, default=False)
    parser.add_argument('--is_trainable', type=bool, default=True,
                        help='Whether the LoRA adapter should be trainable or not.')

    parser.add_argument('--user_name', type=str, default='kxzhang2000', help='wandb register parameter')
    parser.add_argument('--project', type=str, default='Pro_func', help='wandb project name')
    parser.add_argument('--group', type=str, default='NER_V2', help='wandb group')

    return parser.parse_args()


def worker():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"
    args = register_parameters()

    # prepare dataset
    train_files = []
    with open(args.train_data_path, 'r') as file:
        for line in file.readlines():
            train_files.extend(line.strip().split(' '))
    random.seed(args.seed)
    random.shuffle(train_files)

    test_files = []
    with open(args.test_data_path, 'r') as file:
        for line in file.readlines():
            test_files.extend(line.strip().split(' '))


    # initialize trainer class
    trainer = ProteinNERTrainer(output_home=args.output_home, k=args.k)

    # register dataset
    trainer.register_dataset(
        data_files=train_files[:10],
        mode='train',
        dataset_type='class',
        batch_size=args.batch_size,
        model_name_or_path=args.model_path_or_name
    )

    trainer.register_dataset(
        data_files=test_files[:10],
        mode='test',
        dataset_type='class',
        batch_size=args.batch_size,
        model_name_or_path=args.model_path_or_name
    )

    model = ProtTransT5ForAAClassifier(
        model_name_or_path=args.model_path_or_name,
        num_classes=args.num_classes + 1 if args.add_background else args.num_classes,
        lora_inference_mode=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.01,
    )
    trainer.register_model(
        model=model,
        reuse=args.reuse,
        is_trainable=args.is_trainable,
        learning_rate=args.learning_rate
    )

    trainer.train(**vars(args))
    trainer.inference(**vars(args))


if __name__ == '__main__':
    worker()
