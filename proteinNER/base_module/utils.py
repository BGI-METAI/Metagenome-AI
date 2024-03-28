#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 2:51 PM
# @Author  : zhangchao
# @File    : utils.py
# @Email   : zhangchao5@genomics.cn
import logging
import numpy as np


class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = np.Inf

    def __call__(self, loss):
        if np.isnan(loss):
            return True

        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def init_logger(timestamp):
    logging.basicConfig(
        format="%(name)-12s %(levelname)-8s %(message)s",
        level=logging.INFO,
        filename=f"{timestamp}.log",
    )
    return logging.getLogger(__name__)

