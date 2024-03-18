#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/15 14:26
# @Author  : zhangchao
# @File    : parse_configuration.py
# @Email   : zhangchao5@genomics.cn
import argparse

class ParseConfig:
    @staticmethod
    def register_parameters(**kwargs):
        parser = argparse.ArgumentParser(description='Protein Annotation Framework')
        parser.add_argument(
            '--data_path',
            type=str,
            default='/home/share/huadjyin/home/s_cenweixuan/dataset/pfam/sub_pfam',
            help='the path of input dataset'
        )
        parser.add_argument('--seed', type=int, default=42, help='random seed')

        parser.add_argument(
            '--model_path_or_name',
            type=str,
            default='/home/share/huadjyin/home/s_cenweixuan/weight/prot_t5_xl_half_uniref50-enc',
            help='pretrianed model path or name'
        )
        parser.add_argument('--embed_mode', type=str, default='t5', help='which embedding model to be choose,  support `t5`, `bert`, `albert`')
        parser.add_argument('--do_lower_case', type=bool, default=False, help='parameter of pretrained pLMs tokenizer')
        parser.add_argument('--legacy', type=bool, default=False, help='parameter of pretrained pLMs tokenizer')

        parser.add_argument('--train_size', type=float, default=0.7, help='the size of training dataset')
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--num_classes', type=int, default=20794, help='the number of categories')

        parser.add_argument('--local_rank', type=int, default=0, help='Rank of the current process (it should be a number between 0 and ``world_size``-1).')
        parser.add_argument('--world_size', type=int, default=4, help='parameter of initialize DDP')

        return parser.parse_args()






