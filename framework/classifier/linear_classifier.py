#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:10 PM
# @Author  : zhangchao
# @File    : linear_classifier.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn


class FullLinearClassifier(nn.Module):
    """A classification head used to output protein family (or other targets) probabilities

    ****************************
    original implement by Nikola

    2024.03.08 rebuilt by zhangchao
    ****************************

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, num_classes, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            # Single fully connected layer for classification head
            self.classifier = nn.Sequential(nn.Linear(d_model, num_classes))
        # Multiple hidden layers followed by a linear layer for classification head
        else:
            layers = []
            prev_size = d_model
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_classes))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)
