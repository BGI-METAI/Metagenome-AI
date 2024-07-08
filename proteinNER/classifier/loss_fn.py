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

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss