#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 4/10/24 2:21 PM
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import torch
import wandb
import torch.distributed as dist
from tqdm import tqdm

from proteinNER.base_module import BaseTrainer
from proteinNER.base_module.utils import EarlyStopper


class ProtT5EmbeddingIncrementalTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ProtT5EmbeddingIncrementalTrainer, self).__init__(**kwargs)

    def train(self, **kwargs):
        max_epoch = kwargs.get('epoch', 100)
        loss_weight = kwargs.get('loss_weight', 1.)
        patience = kwargs.get('patience', 4)
        load_best_model = kwargs.get('load_best_model', True)
        wandb_username = kwargs.get('user_name')
        wandb_project = kwargs.get('project')
        wandb_group = kwargs.get('group')

        early_stopper = EarlyStopper(patience=patience)
        self.register_wandb(
            user_name=wandb_username,
            project_name=wandb_project,
            group=wandb_group
        )
        wandb.config = {
            "learning_rate": self.learning_rate,
            "max_epoch": max_epoch,
            "batch_size": self.batch_size,
            "loss_weight": loss_weight
        }

        for eph in range(max_epoch):
            self.model.train()
            self.train_loader.sampler.set_epoch(eph)
            batch_iterator = tqdm(self.train_loader, desc=f'Eph: {eph:03d}')
            for idx, sample in enumerate(batch_iterator):
                input_ids, attention_mask = sample
                with torch.cuda.amp.autocast():
                    logist = self.model(input_ids.cuda(), attention_mask.cuda())
                    raise NotImplementedError

    def save_ckpt(self, mode):
        if dist.get_rank() == 0:
            trainer_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict()
            }
            if mode == 'batch':
                torch.save(trainer_dict, osp.join(self.batch_ckpt_home, 'trainer.bin'))
                self.model.module.lora_embedding.save_pretrained(self.batch_ckpt_home)
            elif mode in ['epoch', 'best']:
                torch.save(trainer_dict, osp.join(self.best_ckpt_home, 'trainer.bin'))
                self.model.module.lora_embedding.save_pretrained(self.best_ckpt_home)
            else:
                raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

    def load_ckpt(self, mode, is_trainable=False):
        pass

    @torch.no_grad()
    def inference(self, **kwargs):
        pass
