#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:10 PM
# @Author  : zhangchao
# @File    : protein_aa_ner.py
# @Email   : zhangchao5@genomics.cn
from framework.classifier.linear_layer_module import LinearLayerModule


class AminoAcidsNERClassifier:
    def __init__(self, input_dims, hidden_dims, num_classes):
        self.layer = LinearLayerModule(
            input_dims=input_dims,
            output_dims=hidden_dims,
            add_bias=True,
            is_bn=True,
            is_act=True,
            drop_rate=0.2
        )
        self.classifier_layer = LinearLayerModule(
            input_dims=hidden_dims,
            output_dims=num_classes,
            add_bias=True,
            is_bn=False,
            is_act=False,
            drop_rate=None
        )

    def forward(self, x):
        return self.classifier_layer(self.layer(x))

