#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 5:25 PM
# @Author  : zhangchao
# @File    : loss_fn.py
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn.functional as F


class ProteinLoss:
    @staticmethod
    def cross_entropy_loss(pred, target, weight=1.):
        return F.cross_entropy(pred, target.long().to(pred.device)) * weight

    @staticmethod
    def focal_loss(pred, target, weight=1., gamma=2.):
        ce_loss = F.cross_entropy(pred, target.long().to(pred.device))
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** gamma * ce_loss * weight
        return focal_loss
