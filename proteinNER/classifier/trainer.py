#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:12
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import socket
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
        lr = kwargs.get('lr', 1e-3)
        max_epoch = kwargs.get('epoch', 100)
        loss_weight = kwargs.get('loss_weight', 1.)
        patience = kwargs.get('patience', 4)
        load_best_model = kwargs.get('load_best_model', True)

        wandb_username = kwargs.get('username')
        wandb_project = kwargs.get('project')
        wandb_group = kwargs.get('group')

        reuse = kwargs.get('reuse')
        if not reuse:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)
            self.scaler = torch.cuda.amp.GradScaler()

        early_stopper = EarlyStopper(patience=patience)
        self.register_wandb(
            user_name=wandb_username,
            project_name=wandb_project,
            group=wandb_group
        )
        wandb.config = {
            "learning_rate": lr,
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
                input_ids, attention_mask, batch_label, batch_path = sample
                try:
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

                    # dist.barrier()
                    # gather_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
                    # dist.all_gather(gather_loss, loss)
                    # gathered_mean_loss = torch.stack(gather_loss).mean().item()
                    # batch_iterator.set_postfix({'Loss': f'{gathered_mean_loss:.4f}'})
                    # eph_loss.append(gathered_mean_loss)
                    # wandb.log({'loss': gathered_mean_loss})

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.save_in_batch and idx % 10 == 0:
                        self.save_ckpt('batch')
                except Exception as e:
                    with open(osp.join(self.except_home, f'{socket.gethostname()}.except'), 'a') as except_file:
                        for name in batch_path:
                            except_file.write(name)
                        except_file.write(e)
                        except_file.write(f' split line '.center(100, '*'))
            self.lr_scheduler.step()
            self.valid_model_performance()
            if early_stopper(np.mean(eph_loss)):
                print(f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")
                print(f'The best `ckpt` file has saved in {self.best_ckpt_home}')
                if load_best_model:
                    self.load_ckpt(mode='best')
                break
            elif early_stopper.counter == 0:
                self.save_ckpt(mode='best')

    @torch.no_grad()
    def valid_model_performance(self):
        self.model.eval()
        accuracy_total = []
        for sample in self.test_loader:
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids.cuda(), attention_mask.cuda())
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)
            accuracy = torch.eq(pred, batch_label.to(pred.device)).float().mean()

            dist.barrier()
            gathered_acc = [torch.zeros_like(accuracy) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_acc, accuracy)
            gathered_mean_acc = torch.stack(gathered_acc).mean().item()
            accuracy_total.append(gathered_mean_acc)
        wandb.log({'Accuracy': np.mean(accuracy_total)})

    def inference(self, **kwargs):
        pass
