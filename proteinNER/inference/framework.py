#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/10/24 2:21 PM
# @Author  : zhangchao
# @File    : framework.py
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import T5EncoderModel

from collections import Counter
from typing import List


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.transition = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3,),
            stride=(1,),
            padding='same'
        )
        self.crf = CRF(num_tags=out_channels, batch_first=True)

    def forward(self, embeddings, attention_mask) -> List[dict]:
        with torch.no_grad():
            emissions = self.transition(embeddings.permute(0, 2, 1)).permute(0, 2, 1)
            predict = self.crf.decode(emissions, attention_mask.byte())
            output = self.post_process(emissions, predict)
        return output

    def post_process(self, emissions, predict):
        probability_matrix = (emissions / 0.2).softmax(dim=-1)
        emissions_pred = probability_matrix.argmax(dim=-1)
        output = []
        for idx, pred in enumerate(predict):
            prob = {}
            t_pred = torch.tensor(pred, device=self.crf.transitions.device, requires_grad=False).unsqueeze(1)
            row, col = torch.nonzero(t_pred, as_tuple=True)
            uniq_tag = torch.unique(t_pred[row])
            if 0 < uniq_tag.size(0) <= 2:
                tmp = []
                for x, y in zip(row, t_pred[row]):
                    tmp.append(probability_matrix[idx][x, y])
                prob[f'{uniq_tag[0].item()}'] = torch.tensor(
                    tmp, device=t_pred.device, requires_grad=False).mean().item()
                emissions_label = emissions_pred[idx][:t_pred.size(0)].detach().cpu().tolist()
            elif uniq_tag.size(0) == 0:
                prob['0'] = 0.
                emissions_label = pred
            else:
                statistics = Counter(pred)
                if 0 in statistics.keys():
                    del statistics[0]
                for key, val in statistics.items():
                    if val < 50: continue
                    tmp = []
                    row, col = torch.where(t_pred == key)
                    for x, y in zip(row, t_pred[row]):
                        tmp.append(probability_matrix[idx][x, y])
                    prob[f'{key}'] = torch.tensor(tmp, device=t_pred.device, requires_grad=False).mean().item()
                if len(prob) == 0:
                    prob['0'] = 0.
                emissions_label = emissions_pred[idx][:t_pred.size(0)].detach().cpu().tolist()

            output.append({
                'crf_label': pred,
                'emission_label': emissions_label,
                'probability': prob
            })
        return output


class ProtTrans4AAClassifier(nn.Module):
    def __init__(self, base_model_name_or_path, num_classes, n_header):
        super(ProtTrans4AAClassifier, self).__init__()
        self.base_embedding = T5EncoderModel.from_pretrained(base_model_name_or_path)
        self.multiheader = nn.ModuleList()
        for _ in range(n_header):
            self.multiheader.append(Transition(
                in_channels=self.base_embedding.config.d_model,
                out_channels=num_classes
            ))
        self.disable_grad()

    def disable_grad(self):
        for layer in [self.base_embedding, self.multiheader]:
            for k, v in layer.named_parameters():
                v.requires_grad = False

    def forward(self, input_ids, attention_mask) -> List[dict]:
        results = []
        with torch.no_grad():
            embeddings = self.base_embedding(input_ids, attention_mask).last_hidden_state
            for idx, header in enumerate(self.multiheader):
                output = header(embeddings, attention_mask)
                results.append(output)

        return results
