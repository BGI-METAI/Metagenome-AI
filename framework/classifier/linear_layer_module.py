#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/12/24 10:07 AM
# @Author  : zhangchao
# @File    : linear_layer_module.py
# @Email   : zhangchao5@genomics.cn
from typing import Optional, Union

import torch.nn as nn


class LinearLayerModule(nn.Module):
    def __init__(
            self,
            input_dims: int,
            output_dims: int,
            add_bias: bool = True,
            is_bn: bool = True,
            is_act: bool = True,
            drop_rate: Optional[Union[None, float]] = None
    ):
        super(LinearLayerModule, self).__init__()
        if add_bias:
            self.layer = nn.Linear(input_dims, output_dims)
        else:
            self.layer = nn.Linear(input_dims, output_dims, bias=False)

        if is_bn:
            self.bn = nn.LayerNorm(output_dims)
        else:
            self.bn = None

        if is_act:
            self.act = nn.ReLU()
        else:
            self.act = None

        if drop_rate:
            if not drop_rate <= 0 and 1. > drop_rate:
                self.drop = nn.Dropout(p=drop_rate)
            else:
                self.drop = None
        else:
            self.drop = None

    def forward(self, x):
        output = self.layer(x)
        if self.bn:
            output = self.bn(output)
        if self.act:
            output = self.act(output)
        if self.drop:
            output = self.drop(output)
        return output
