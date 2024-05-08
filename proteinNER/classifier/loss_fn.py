#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 5:25 PM
# @Author  : zhangchao
# @File    : loss_fn.py
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn.functional as F
from torch import nn


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


class MultiTaskLoss(nn.Module):
    """ Weighs multiple loss functions by considering the
        homoscedastic uncertainty of each task """

    def __init__(self, device="cpu"):
        self.device = device
        super(MultiTaskLoss, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros((6)))

    def mse(self, outputs, labels, mask):
        loss = torch.square(outputs - labels) * mask
        return torch.sum(loss) / torch.sum(mask)

    def cross_entropy(self, outputs, labels, mask):
        labels = labels.clone()
        labels[mask == 0] = -999

        return nn.CrossEntropyLoss(ignore_index=-999)(outputs, labels.long())

    def ss8(self, outputs, labels, mask):
        labels = torch.argmax(labels[:, :, 7:15], dim=2)
        outputs = outputs[0].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def ss3(self, outputs, labels, mask):
        structure_mask = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]).to(self.device)

        labels = torch.max(labels[:, :, 7:15] * structure_mask, dim=2)[0].long()
        outputs = outputs[1].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def disorder(self, outputs, labels, mask):
        # apply the disorder loss
        labels = labels[:, :, 1].unsqueeze(2)
        labels = torch.argmax(torch.cat([labels, 1 - labels], dim=2), dim=2)

        outputs = outputs[2].permute(0, 2, 1)

        return self.cross_entropy(outputs, labels, mask)

    def rsa(self, outputs, labels, mask):
        labels = labels[:, :, 5].unsqueeze(2)
        outputs = outputs[3]

        mask = mask.unsqueeze(2)

        return self.mse(outputs, labels, mask)

    def phi(self, outputs, labels, mask):
        labels = labels[:, :, 15].unsqueeze(2)
        outputs = outputs[4]

        mask = mask * (labels != 360).squeeze(2).int()
        mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

        loss = self.mse(outputs.squeeze(2),
                        torch.cat(
                            (torch.sin(self.dihedral_to_radians(labels)), torch.cos(self.dihedral_to_radians(labels))),
                            dim=2).squeeze(2), mask)
        return loss

    def psi(self, outputs, labels, mask):
        labels = labels[:, :, 16].unsqueeze(2)
        outputs = outputs[5]

        mask = mask * (labels != 360).squeeze(2).int()
        mask = torch.cat(2 * [mask.unsqueeze(2)], dim=2)

        loss = self.mse(outputs.squeeze(2),
                        torch.cat(
                            (torch.sin(self.dihedral_to_radians(labels)), torch.cos(self.dihedral_to_radians(labels))),
                            dim=2).squeeze(2), mask)
        return loss

    def forward(self, outputs, labels, weighted=True):
        """ Forwarding of the multitaskloss input
        Args:
            outputs (torch.tensor): output data from model
            labels (torch.tensor): corresponding labels for the output
        """

        # filters
        zero_mask = labels[:, :, 0]
        disorder_mask = labels[:, :, 1]
        unknown_mask = labels[:, :, -1]

        # weighted losses
        ss8 = self.ss8(outputs, labels, zero_mask) * 1
        ss3 = self.ss3(outputs, labels, zero_mask) * 5
        dis = self.disorder(outputs, labels, zero_mask) * 5
        rsa = self.rsa(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 100
        phi = self.phi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5
        psi = self.psi(outputs, labels, zero_mask * disorder_mask * unknown_mask) * 5

        loss = torch.stack([ss8, ss3, dis, rsa, phi, psi])

        return loss.sum()

    @staticmethod
    def dihedral_to_radians(angle):
        """ Converts angles to radians
        Args:
            angles (1D Tensor): vector with angle values
        """
        return angle * np.pi / 180
