#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : diffusion_trainer.py
# @Author  : zhangchao
# @Date    : 2024/6/26 17:39 
# @Email   : zhangchao5@genomics.cn
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from dataclasses import field

from proteinNER.diffusion.diffusion import SimpleDiffusion
from proteinNER.diffusion.schedule import BetaSchedule
from proteinNER.diffusion.sequence_encode import ProteinSequenceEmbedding, SequenceLabelEmbedding
from proteinNER.diffusion.stack_atten import ProteinFuncAttention


class DiffusionProteinFuncModel(nn.Module):
    def __init__(
            self,
            model_name_or_path,
            num_labels,
            betas,
            n_header=16,
    ):
        super(DiffusionProteinFuncModel, self).__init__()

        self.protein_module = ProteinSequenceEmbedding(model_name_or_path)
        self.label_module = SequenceLabelEmbedding(
            num_embeddings=num_labels,
            embedding_dim=self.protein_module.embedding.config.d_model
        )

        self.diffusion = SimpleDiffusion(betas=betas)
        self.denoise_model = ProteinFuncAttention(
            d_model=self.protein_module.embedding.config.d_model,
            n_header=n_header
        )

    def forward(self, input_ids, attention_mask, labels, timestep):
        embed_seq = self.protein_module(input_ids, attention_mask)
        embed_label = self.label_module(labels)

        x_start = torch.cat((embed_seq, embed_label), dim=1)
        std = self.diffusion.extract_into_tensor(
            self.diffusion.sqrt_one_minus_alphas_cumprod,
            torch.tensor([0]).to(x_start.device),
            x_start.shape
        )
        x_start + torch.randn_like(x_start) * std
        x_start[:, :embed_seq.size(1), :] = embed_seq
        x_t = self.diffusion.q_sample(x_start, timestep)

        model_output = self.denoise_model(x_t, timestep)
        return model_output


class DiffusionProteinFuncTrainer:
    diffusion: SimpleDiffusion = field(default=None, metadata={'help': 'gaussian diffusion model'})
    sequence_model: torch.nn.Module = field(default=None, metadata={'help': 'embedding protein sequence model'})
    denoise_model: torch.nn.Module = field(default=None, metadata={'help': 'denoise model'})
    data_loader: DataLoader = field(default=None, metadata={"help": "data loader"})

    def __init__(self, **kwargs):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    from proteinNER.base_module import CustomNERDataset
    from functools import partial

    model_name_or_path = '/home/share/huadjyin/home/zhangchao5/weight/prot_t5_xl_half_uniref50-enc'
    data_path = '/home/share/huadjyin/home/zhangchao5/dataset/version2/pfam/filtered/less250/subgroup/sub500.01/sub500.01.train.txt'
    label2id = '/home/share/huadjyin/home/zhangchao5/dataset/version2/pfam/record/diffusion/diffusion_label2id.pkl'

    data_files = []
    with open(data_path, 'r') as fp:
        for line in fp.readlines():
            data_files.append(line.strip())
    dataset = CustomNERDataset(
        processed_sequence_label_pairs_path=data_files,
        label2id_path=label2id,
        tokenizer_model_name_or_path=model_name_or_path,
    )

    data_loader = DataLoader(dataset, batch_size=10, collate_fn=partial(dataset.collate_fn, is_valid=False))

    for sample in data_loader:
        input_ids, attention_mask, batch_label = sample
        print()

    betas = BetaSchedule(num_timestep=200).cosine_beta_schedule()
    model = DiffusionProteinFuncModel(
        model_name_or_path=model_name_or_path,
        num_labels=500,
        betas=betas,
    )

