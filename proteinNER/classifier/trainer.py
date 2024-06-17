#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:12
# @Author  : zhangchao
# @File    : framework.py
# @Email   : zhangchao5@genomics.cn
import torch
import pickle
import os.path as osp
import numpy as np

from tqdm import tqdm
from datetime import datetime

from proteinNER.base_module import BaseTrainer
from proteinNER.base_module.utils import EarlyStopper

CKPT_SAVE_STEP = 10


class ProteinAANERTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(ProteinAANERTrainer, self).__init__(**kwargs)

    def train(self, **kwargs):
        self.loss_weight = kwargs.get('loss_weight', 1.)

        early_stopper = EarlyStopper(patience=kwargs.get('patience', 4))
        self.register_wandb(
            user_name=kwargs.get('username'),
            project_name=kwargs.get('project'),
            group=kwargs.get('group')
        )

        self.model = self.accelerator.prepare_model(self.model)
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.train_loader = self.accelerator.prepare_data_loader(self.train_loader)
        self.lr_scheduler = self.accelerator.prepare_scheduler(self.lr_scheduler)

        for eph in range(kwargs.get('epoch', 100)):
            self.model.train()
            batch_iterator = tqdm(self.train_loader,
                                  desc=f'Pid: {self.accelerator.process_index} Eph: {eph:03d} ({early_stopper.counter} / {early_stopper.patience})')
            eph_loss = []
            for idx, sample in enumerate(batch_iterator):
                input_ids, attention_mask, batch_label = sample
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        loss = self.model(input_ids, attention_mask, batch_label.long())
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                self.accelerator.log({'loss': loss.item()})
                eph_loss.append(loss.item())
                if self.save_in_batch and idx % CKPT_SAVE_STEP == 0:
                    self.save_ckpt('batch')

                with self.accelerator.main_process_first():
                    self.accelerator.log({'learning rate': self.optimizer.state_dict()['param_groups'][0]['lr']})
            self.lr_scheduler.step()

            if early_stopper(np.mean(eph_loss)):
                if np.isnan(np.mean(eph_loss)):
                    self.accelerator.print(
                        f"\n{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model training ended unexpectedly!")

                self.accelerator.print(
                    f"\nPid: {self.accelerator.process_index}: {datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")

                self.accelerator.wait_for_everyone()
                with self.accelerator.main_process_first():
                    self.accelerator.print(
                        f'\n\nPid: {self.accelerator.process_index}: The best `ckpt` file has saved in {self.best_ckpt_home}')
                self.accelerator.end_training()
                break
            elif early_stopper.counter == 0:
                self.save_ckpt(mode='best')

    @torch.no_grad()
    def valid_model_performance(self, **kwargs):
        self.inference(*kwargs)

    @torch.no_grad()
    def inference(self, **kwargs):
        model = self.accelerator.prepare_model(self.model)
        data_loader = self.accelerator.prepare_data_loader(self.test_loader)

        self.model.eval()

        batch_iterator = tqdm(data_loader, desc=f'Pid: {self.accelerator.process_index}')

        for idx, sample in enumerate(batch_iterator):
            input_ids, attention_mask, batch_label, batch_protein_ids = sample
            results = model.module.inference(input_ids, attention_mask)

            for cnt in range(self.batch_size):
                results[cnt].update({'ground_label': batch_label[cnt].detach().cpu().tolist()})
                pickle.dump(results[cnt], open(osp.join(self.result_home, f'{batch_protein_ids[cnt]}.pkl'), 'wb'))
