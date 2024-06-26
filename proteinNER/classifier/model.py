#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 9:47
# @Author  : zhangchao
# @File    : model.py
# @Email   : zhangchao5@genomics.cn
from typing import List

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import T5EncoderModel
from collections import Counter


class TransitionModel(nn.Module):
    """
    Transition Model

    Args:
        in_channels: int, the number of input channels
        hidden_channels: int, the number of hidden channels
        out_channels: int, the number of output channels

    Returns:
        output: torch.Tensor, the output tensor
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TransitionModel, self).__init__()
        raise NotImplementedError


class ProtT5Conv1dCRF4AAClassifier(nn.Module):
    """
    Protein T5 Conv1d CRF Classifier

    Args:
        model_name_or_path: str, the pre-trained model name or path
        num_classes: int, the number of classes

    Returns:
        loss: torch.Tensor, the negative log likelihood loss

    Examples:
        model = ProtT5Conv1dCRF4AAClassifier(model_name_or_path='Rostlab/prot_t5_xl_uniref50', num_classes=21)
        input_ids = torch.randint(0, 20, (2, 512))
        attention_mask = torch.ones(2, 512)
        labels = torch.randint(0, 21, (2, 512))
        loss = model(input_ids, attention_mask, labels)
    """
    def __init__(self, model_name_or_path, num_classes):
        super(ProtT5Conv1dCRF4AAClassifier, self).__init__()
        self.base_embedding = T5EncoderModel.from_pretrained(model_name_or_path)
        for k, v in self.base_embedding.named_parameters():
            v.requires_grad = False

        self.transition = nn.Conv1d(
            in_channels=self.base_embedding.config.d_model,
            out_channels=num_classes,
            kernel_size=(3,),
            stride=(1,),
            padding='same'
        )
        self.crf = CRF(num_tags=num_classes, batch_first=True)

    def forward(self, input_ids, attention_mask, labels):
        with torch.no_grad():
            embeddings = self.base_embedding(input_ids, attention_mask).last_hidden_state
        embeddings = self.transition(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        loss = -1 * self.crf(embeddings, labels, mask=attention_mask.byte())
        return loss

    @torch.no_grad()
    def inference(self, input_ids, attention_mask) -> List[dict]:
        embeddings = self.base_embedding(input_ids, attention_mask).last_hidden_state
        emissions = self.transition(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
        predict = self.crf.decode(emissions, attention_mask.byte())
        output = self.post_process(emissions, predict)
        return output

    def post_process(self, emissions, predict) -> List[dict]:
        probability_matrix = (emissions/0.01).softmax(dim=-1)
        emissions_pred = probability_matrix.argmax(dim=-1)
        output = []

        for idx, pred in enumerate(predict):
            prob = {}
            t_pred = torch.tensor(pred, device=self.base_embedding.device, requires_grad=False).unsqueeze(1)
            row, col = torch.nonzero(t_pred, as_tuple=True)
            uniq_tag = torch.unique(t_pred[row])
            if 0 < uniq_tag.size(0) <= 2:
                tmp = []
                for x, y in zip(row, t_pred[row]):
                    tmp.append(probability_matrix[idx][x, y])
                prob[f'{uniq_tag[0].item()}'] = torch.tensor(tmp, device=t_pred.device, requires_grad=False).mean().item()
                emissions_label = emissions_pred[idx][:t_pred.size(0)].detach().cpu().tolist()
            elif uniq_tag.size(0) == 0:
                prob['0'] = 0.
                emissions_label = pred
            else:
                statistics = Counter(pred)
                if 0 in statistics.keys():
                    del statistics[0]
                for key, val in statistics.items():
                    if val < 5: continue
                    tmp = []
                    row, col = torch.where(t_pred == key)
                    for x, y in zip(row, t_pred[row]):
                        tmp.append(probability_matrix[idx][x, y])
                    prob[f'{key}'] = torch.tensor(tmp, device=t_pred.device, requires_grad=False).mean().item()
                emissions_label = emissions_pred[idx][:t_pred.size(0)].detach().cpu().tolist()

            output.append({
                'crf_label': pred,
                'emission_label': emissions_label,
                'probability': prob
            })
        return output


class DiscriminatorLayer(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(DiscriminatorLayer, self).__init__()
        self.ly = nn.Linear(input_dims, output_dims)
        self.bn = nn.BatchNorm1d(output_dims)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.ly(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x


class ProteinDiscriminator(nn.Module):
    """
    Protein Discriminator

    Args:
        input_dims: int, input length of the predicted protein sequence

    Returns:
        output: torch.Tensor, the protein sequence is real or fake
    """
    def __init__(self, input_dims):
        super(ProteinDiscriminator, self).__init__()
        self.net = nn.ModuleList()
        for _ in range(4):
            self.net.append(
                DiscriminatorLayer(input_dims=input_dims, output_dims=input_dims)
            )
        self.out = nn.Linear(input_dims, 2)

    def forward(self, x):
        for idx, ly in enumerate(self.net):
            x += ly(x)
        out = self.out(x)
        return out
