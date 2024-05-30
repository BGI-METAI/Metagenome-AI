#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/30 15:18
# @Author  : zhangchao
# @File    : demo_prottrans_amino_acids_classifier_inference.py
# @Email   : zhangchao5@genomics.cn
import argparse

from proteinNER.inference.infer import ProtTransAminoAcidClassifierInfer


def register_parameters():
    parser = argparse.ArgumentParser(description='Protein Sequence Amino Acids Annotation Framework')
    parser.add_argument('--data_path', type=str, required=True, help='the path of inference dataset')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='pre-trained pLMs model name or path')
    parser.add_argument('--amino_acid_classifier_model_path', type=str, required=True, help='the saved amino acid header')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_classes', type=int, required=True, help='the number of amino acid categories')
    parser.add_argument('--num_header', type=int, required=True, help='the number of classifiers')

    parser.add_argument('--user_name', type=str, default='zhangchao162', help='wandb register parameter')
    parser.add_argument('--project', type=str, default='ProteinConvCRF', help='wandb project name')
    parser.add_argument('--group', type=str, default='accelerate', help='wandb group')

    return parser.parse_args()


def worker():
    args = register_parameters()

    data_files = []
    with open(args.data_path, 'r') as fp:
        for line in fp.readlines():
            data_files.append(line.strip())

    infer = ProtTransAminoAcidClassifierInfer(pairs_files=data_files, **vars(args))
    infer.inference()


if __name__ == '__main__':
    worker()

