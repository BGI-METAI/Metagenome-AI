#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:10 PM
# @Author  : zhangchao
# @File    : protein_aa_ner.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn
from framework.classifier.linear_layer_module import LinearLayerModule


class AminoAcidsNERClassifier(nn.Module):
    def __init__(self, input_dims, num_classes):
        super(AminoAcidsNERClassifier, self).__init__()
        self.classifier_layer = LinearLayerModule(
            input_dims=input_dims,
            output_dims=num_classes,
            add_bias=False,
            is_bn=False,
            is_act=False,
            drop_rate=None
        )

    def forward(self, x):
        return self.classifier_layer(x)

