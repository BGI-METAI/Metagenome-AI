#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:12
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import wandb
import torch
import torch.distributed as dist
import numpy as np

from tqdm import tqdm
from datetime import datetime

from proteinNER.base_module import BaseTrainer
from proteinNER.base_module.utils import EarlyStopper
from proteinNER.classifier.loss_fn import ProteinLoss


class ProteinNERTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ProteinNERTrainer, self).__init__(**kwargs)

    def train(self, **kwargs):
        max_epoch = kwargs.get('epoch', 100)
        loss_weight = kwargs.get('loss_weight', 1.)
        patience = kwargs.get('patience', 4)
        load_best_model = kwargs.get('load_best_model', True)

        wandb_username = kwargs.get('username')
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
            eph_loss = []
            self.model.train()
            self.train_loader.sampler.set_epoch(eph)
            batch_iterator = tqdm(self.train_loader, desc=f'Eph: {eph:03d}')
            for idx, sample in enumerate(batch_iterator):
                input_ids, attention_mask, batch_label = sample
                with torch.cuda.amp.autocast():
                    logist = self.model(input_ids.cuda(), attention_mask.cuda())
                    loss = ProteinLoss.focal_loss(
                        pred=logist.permute(0, 2, 1),
                        target=batch_label,
                        weight=loss_weight,
                        gamma=2.
                    )

                batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                eph_loss.append(loss.item())
                wandb.log({'loss': loss.item()})

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.save_in_batch and idx % 10 == 0:
                    self.save_ckpt('batch')
            self.lr_scheduler.step()
            self.valid_model_performance()
            if early_stopper(np.mean(eph_loss)):
                print(f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")
                print(f'The best `ckpt` file has saved in {self.best_ckpt_home}')
                if load_best_model:
                    self.load_ckpt(mode='best', is_trainable=False)
                break
            elif early_stopper.counter == 0:
                self.save_ckpt(mode='best')

    @staticmethod
    def distributed_concat(tensor, num_total_examples):
        output_tensor = [tensor.clone() for _ in range(dist.get_world_size())]
        dist.all_gather(output_tensor, tensor)
        concat = torch.cat(output_tensor, dim=0)
        return concat[:num_total_examples]

    @torch.no_grad()
    def valid_model_performance(self):
        self.model.eval()
        total_tag = []
        for sample in self.test_loader:
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids.cuda(), attention_mask.cuda())
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)
            correct = torch.eq(pred, batch_label.to(pred.device)).float().sum(1)
            tag = (correct == pred.shape[-1]).float()
            total_tag.append(tag)
        total_tag = self.distributed_concat(torch.concat(total_tag, dim=0), len(self.test_loader))
        accuracy = total_tag.sum() / len(total_tag)
        wandb.log({'Accuracy': np.mean(accuracy.item())})

    @torch.no_grad()
    def inference(self, **kwargs):
        pass
