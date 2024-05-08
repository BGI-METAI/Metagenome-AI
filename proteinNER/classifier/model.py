#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 9:47
# @Author  : zhangchao
# @File    : model.py
# @Email   : zhangchao5@genomics.cn
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import T5EncoderModel, T5Model


class ProtTransT5EmbeddingPEFTModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1
    ):
        super(ProtTransT5EmbeddingPEFTModel, self).__init__()
        self.base_model = T5EncoderModel.from_pretrained(model_name_or_path)
        peft_config = LoraConfig(
            inference_mode=lora_inference_mode,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.lora_embedding = get_peft_model(self.base_model, peft_config)

    def forward(self, input_ids, attention_mask):
        return self.lora_embedding(input_ids, attention_mask).last_hidden_state


class ProtTransT5ForAAClassifier(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_classes,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
    ):
        super(ProtTransT5ForAAClassifier, self).__init__()
        self.embedding = ProtTransT5EmbeddingPEFTModel(
            model_name_or_path=model_name_or_path,
            lora_inference_mode=lora_inference_mode,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.classifier = nn.Linear(self.embedding.lora_embedding.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.embedding(input_ids, attention_mask))


class ProtTransT5MaskPEFTModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_classes,
            lora_inference_mode=False,
            lora_r=8,
            lora_alpha=32,
            lora_dropout=0.1,
    ):
        super(ProtTransT5MaskPEFTModel, self).__init__()
        self.embedding = ProtTransT5EmbeddingPEFTModel(
            model_name_or_path=model_name_or_path,
            lora_inference_mode=lora_inference_mode,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.classifier = nn.Linear(self.embedding.lora_embedding.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.embedding(input_ids, attention_mask))


class ProTransEmbeddingModel(nn.Module):
    def __init__(self, lora_embedding, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lora_embedding = lora_embedding

    def forward(self, input_ids, attention_mask):
        return self.lora_embedding(input_ids, attention_mask).last_hidden_state


class ProtTransT5MaskPretrainModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_classes,
    ):
        super(ProtTransT5MaskPretrainModel, self).__init__()
        self.base_model = T5EncoderModel.from_pretrained(model_name_or_path)
        self.embedding = ProTransEmbeddingModel(lora_embedding=self.base_model)
        self.classifier = nn.Linear(self.embedding.lora_embedding.config.d_model, num_classes)

    def forward(self, input_ids, attention_mask):
        return self.classifier(self.embedding(input_ids, attention_mask))


class NetsurfConvModel(nn.Module):
    def __init__(self, n_final_in=32):
        """
        ref: https://github.com/Eryk96/NetSurfP-3.0
        :param n_final_in: dimension for final in layer
        """
        super(NetsurfConvModel, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = nn.Sequential(
            nn.Conv2d(1024, n_final_in, kernel_size=(7, 1), padding=(3, 0)),  # 7x32
            nn.ReLU(),
            nn.Dropout(0.25),
        )
        self.dssp3_classifier = nn.Sequential(
            nn.Conv2d(n_final_in, 3, kernel_size=(7, 1), padding=(3, 0))  # 7
        )

        self.dssp8_classifier = nn.Sequential(
            nn.Conv2d(n_final_in, 8, kernel_size=(7, 1), padding=(3, 0))
        )
        self.diso_classifier = nn.Sequential(
            nn.Conv2d(n_final_in, 2, kernel_size=(7, 1), padding=(3, 0))
        )

    def forward(self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0, 2, 1).unsqueeze(dim=-1)
        x = self.elmo_feature_extractor(x)  # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 8)
        diso_Yhat = self.diso_classifier(x).squeeze(dim=-1).permute(0, 2, 1)  # OUT: (B x L x 2)
        return d3_Yhat, d8_Yhat, diso_Yhat
