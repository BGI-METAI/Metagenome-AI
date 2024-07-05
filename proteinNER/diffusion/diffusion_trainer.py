#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @Project : Metagenome-AI 
# @File    : diffusion_trainer.py
# @Author  : zhangchao
# @Date    : 2024/6/26 17:39 
# @Email   : zhangchao5@genomics.cn
import random
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from proteinNER.base_module.utils import EarlyStopper
from proteinNER.diffusion.denoise_model import DenoiseModel
from proteinNER.diffusion.diffusion import SimpleDiffusion
from proteinNER.diffusion.schedule import BetaSchedule, TimestepScheduleSampler
from proteinNER.diffusion.sequence_encode import ProteinSequenceEmbedding, ProteinLabelEmbedding
from proteinNER.diffusion.stack_atten import ProteinFuncAttention
from proteinNER.base_module import BaseTrainer


class DiffusionProteinModel(nn.Module):
    def __init__(
            self,
            pretrianed_sequence_model_name_or_path,
            base_label_model_name_or_path,
            num_vocabs,
            betas
    ):
        super(DiffusionProteinModel, self).__init__()
        self.protein_seq_module = ProteinSequenceEmbedding(model_name_or_path=pretrianed_sequence_model_name_or_path)
        self.label_seq_module = ProteinLabelEmbedding(
            pretrained_model_name_or_path=base_label_model_name_or_path,
            num_vocabs=num_vocabs)
        self.diffusion = SimpleDiffusion(betas=betas)
        self.denoise_module = DenoiseModel(
            model_name_or_path=pretrianed_sequence_model_name_or_path,
            num_classes=num_vocabs)

    def forward(
            self,
            seq_input_ids,
            seq_attention_mask,
            raw_input_ids,
            raw_attention_mask,
            mask_input_ids,
            mask_attention_mask,
            timestep
    ):
        seq_embedding = self.protein_seq_module(seq_input_ids, seq_attention_mask)
        raw_label_embedding = self.label_seq_module(raw_input_ids, raw_attention_mask)
        mask_label_embedding = self.label_seq_module(mask_input_ids, mask_attention_mask)

        # mask loss
        mask_loss = F.cross_entropy(mask_label_embedding.permute(0, 2, 1), raw_input_ids)
        # contrast loss
        contr_loss = self.contrast_loss(embed_q=raw_label_embedding.detach(), embed_k=mask_label_embedding)

        # forward diffusion
        x_start = torch.cat((seq_embedding, raw_label_embedding), dim=1)
        std = self.diffusion.extract_into_tensor(
            self.diffusion.sqrt_one_minus_alphas_cumprod,
            torch.tensor([0]).to(x_start.device),
            x_start.shape
        )
        x_start += torch.randn_like(x_start) * std
        x_start[:, :raw_label_embedding.size(1), :] = raw_label_embedding
        x_t = self.diffusion.q_sample(x_start=x_start, timestep=timestep)
        model_output = self.denoise_module(x_t, timestep)

    def contrast_loss(self, embed_q, embed_k, tau=0.07):
        assert embed_q.size() == embed_k.size()
        B, L, C = embed_q.size()
        embed_q = embed_q.view(B, -1)
        embed_k = embed_k.view(B, -1)

        sim_matrix = torch.einsum('ik, jk -> ij', embed_q, embed_k) / torch.einsum(
            'i, j -> ij', embed_q.norm(p=2, dim=1), embed_k.norm(p=2, dim=1)
        )
        label = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix / tau, label)
        return loss


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
        self.denoise_module = ProteinFuncAttention(
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

        model_output = self.denoise_module(x_t, timestep)

        # diffusion MSE loss
        mse_loss = self.diffusion.mean_flat((x_start - model_output) ** 2).mean()
        out_mean, _, _ = self.diffusion.q_mean_variance(
            x_start,
            torch.LongTensor([self.diffusion.num_timestep - 1]).to(x_start.device)
        )
        tT_loss = self.diffusion.mean_flat(out_mean ** 2).mean()

        # match loss
        match_loss = 0.
        for i in range(x_start[:, :embed_seq.size(1), :].size(-1)):
            match_loss += self.contrast_loss(
                x_start[:, :embed_seq.size(1), i],
                x_start[:, embed_seq.size(1):, i],
                tau=0.07
            ) / x_start.size(-1)

        # contrast loss
        ctr_loss = 0.
        for i in range(x_start[:, embed_seq.size(1):, :].size(-1)):
            ctr_loss += self.contrast_loss(
                x_start[:, embed_seq.size(1):, i],
                x_start[:, embed_seq.size(1):, i],
                tau=0.07
            ) / x_start.size(-1)

        loss = mse_loss + tT_loss + match_loss + ctr_loss

        return loss

    def contrast_loss(self, feat1, feat2, tau):
        sim_matrix = torch.einsum('ik, jk -> ij', feat1, feat2) / torch.einsum(
            'i, j -> ij', feat1.norm(p=2, dim=1), feat2.norm(p=2, dim=1)
        )
        label = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
        loss = torch.nn.functional.cross_entropy(sim_matrix / tau, label)
        return loss


class DiffusionProteinFuncTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(DiffusionProteinFuncTrainer, self).__init__(**kwargs)
        self.learning_rate = kwargs.get('learning_rate')
        self.batch_size = kwargs.get('batch_size')
        self.is_trainable = kwargs.get('is_trainable')
        self.reuse = kwargs.get('reuse_params')
        self.num_timestep = kwargs.get('num_timestep')
        self.decay_gamma = kwargs.get('decay_gamma')
        self.decay_step = kwargs.get('decay_step')

        self.register_wandb(
            user_name=kwargs.get('user_name'),
            project_name=kwargs.get('project_name'),
            group=kwargs.get('group')
        )

        data_files = []
        with open(kwargs.get('data_path'), 'r') as fp:
            for line in fp.readlines():
                data_files.append(line.strip())
        random.seed(kwargs.get('seed'))
        random.shuffle(data_files)

        self.register_dataset(
            data_files,
            label2id_path=kwargs.get('label2id_path'),
            mode=kwargs.get('mode', 'train'),
            model_name_or_path=kwargs.get('model_name_or_path'),
            batch_size=self.batch_size
        )
        betas = BetaSchedule(num_timestep=self.num_timestep).cosine_beta_schedule()
        model = DiffusionProteinFuncModel(
            model_name_or_path=kwargs.get('model_name_or_path'),
            num_labels=kwargs.get('num_label'),
            betas=betas,
            n_header=16,
        )
        self.register_model(model=model)

    def register_model(self, model, **kwargs):
        self.model = model
        if self.is_trainable:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate
            )

    def custom_lr_scheduler(self, epoch, decay_gamma=0.99, decay_step=100):
        lr = self.learning_rate * (decay_gamma ** (epoch // decay_step))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, patience=5, epoch=100):
        early_stopper = EarlyStopper(patience=patience)
        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.train_loader = self.accelerator.prepare_data_loader(self.train_loader)

        for eph in range(epoch):
            self.model.train()
            batch_iterator = tqdm(
                self.train_loader,
                desc=f'PID: {self.accelerator.process_index} EPH: {eph:03d}'
            )
            epoch_loss = []
            for idx, sample in enumerate(batch_iterator):
                input_ids, attention_mask, batch_label = sample
                timestep, weight = TimestepScheduleSampler(
                    num_timestep=self.num_timestep).sample(batch_size=self.batch_size)
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        loss = self.model(input_ids, attention_mask, labels=batch_label, timestep=timestep)
                        self.accelerator.backward(loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    if idx % 100 == 0:
                        self.save_ckpt('batch')
                    epoch_loss.append(loss.item())
                    batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                    self.accelerator.log({'loss': loss.item()})
                    self.accelerator.log({'learning rate': self.optimizer.state_dict()['param_groups'][0]['lr']})

            self.custom_lr_scheduler(epoch=eph, decay_gamma=self.decay_gamma, decay_step=self.decay_step)
        self.accelerator.end_training()

    def save_ckpt(self, mode):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        trainer_dict = {
            "optimizer": self.optimizer.state_dict(),
        }
        label_module_state_dict = {"state_dict": unwrapped_model.label_module.state_dict()}
        denoise_module_state_dict = {"state_dict": unwrapped_model.denoise_module.state_dict()}

        save_path = self.batch_ckpt_home if mode == 'batch' else self.best_ckpt_home
        self.accelerator.save(label_module_state_dict, osp.join(save_path, 'LabelEmbed.bin'))
        self.accelerator.save(denoise_module_state_dict, osp.join(save_path, 'ProteinDiffusion.bin'))
        self.accelerator.save(trainer_dict, osp.join(save_path, 'trainer.bin'))

    def inference(self, **kwargs):
        pass
