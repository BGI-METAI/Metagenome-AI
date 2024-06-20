#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:12
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import torch
import pickle
import os.path as osp
import numpy as np
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from proteinNER.base_module import BaseTrainer
from proteinNER.base_module.dataset import DiscriminatorDataset
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
            batch_iterator = tqdm(
                self.train_loader,
                desc=f'Pid: {self.accelerator.process_index} \
                Eph: {eph:03d} ({early_stopper.counter} / {early_stopper.patience})')
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

                self.accelerator.wait_for_everyone()
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

            for cnt in range(len(results)):
                results[cnt].update({'ground_label': batch_label[cnt].detach().cpu().tolist()})
                pickle.dump(results[cnt], open(osp.join(self.result_home, f'{batch_protein_ids[cnt]}.pkl'), 'wb'))


class DiscriminatorTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super(DiscriminatorTrainer, self).__init__(**kwargs)

    def register_dataset(
            self,
            data_files,
            mode,
            **kwargs
    ):
        self.batch_size = kwargs.get('batch_size')
        max_length = kwargs.get('max_length')
        dataset = DiscriminatorDataset(data_list=data_files, max_length=max_length)
        if mode == 'train':
            self.train_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.collate_fn)
        elif mode == 'test':
            self.test_loader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=dataset.collate_fn)
        else:
            raise ValueError('Got an invalid data loader mode, ONLY SUPPORT: `train` and `test`!')

    def save_ckpt(self, mode):
        if self.accelerator.main_process_first():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            train_dict = {
                'optimizer': self.optimizer.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict()
            }
            state_dict = unwrapped_model.state_dict()

            if mode == 'batch':
                self.accelerator.save(train_dict, osp.join(self.batch_ckpt_home, 'discriminator_trainer.bin'))
                self.accelerator.save(state_dict, osp.join(self.batch_ckpt_home, 'discriminator.bin'))
            elif mode == 'best':
                self.accelerator.save(train_dict, osp.join(self.best_ckpt_home, 'discriminator_trainer.bin'))
                self.accelerator.save(state_dict, osp.join(self.best_ckpt_home, 'discriminator.bin'))
            else:
                raise ValueError(f'Got an invalid mode: `{mode}`')

    def load_ckpt(self, mode, is_trainable=False):
        if mode == 'batch':
            ckpt_home = self.batch_ckpt_home
        elif mode in ['epoch', 'best']:
            ckpt_home = self.best_ckpt_home
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

        if is_trainable:
            trainer_ckpt = torch.load(osp.join(ckpt_home, 'discriminator_trainer.bin'),
                                      map_location=torch.device('cuda'))
            self.optimizer.load_state_dict(trainer_ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(trainer_ckpt['lr_scheduler'])

        state_dict = self.model.state_dict()

        discriminator_dict = torch.load(osp.join(ckpt_home, 'discriminator.bin'), map_location=torch.device('cuda'))

        self.model.load_state_dict(discriminator_dict)

    def train(self, **kwargs):
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
            batch_iterator = tqdm(
                self.train_loader,
                desc=f'Pid: {self.accelerator.process_index} \
                Eph: {eph:03d} ({early_stopper.counter} / {early_stopper.patience})')
            eph_loss = []
            for idx, sample in enumerate(batch_iterator):
                data, label = sample
                with self.accelerator.accumulate(self.model):
                    with self.accelerator.autocast():
                        predict = self.model(data)
                        loss = F.cross_entropy(input=predict, target=label)
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
                        f"\n{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} \
                        (PID: {self.accelerator.process_index}) Model training ended unexpectedly!")

                self.accelerator.print(
                    f"\nPID: {self.accelerator.process_index}: \
                    {datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")

                with self.accelerator.main_process_first():
                    self.accelerator.print(
                        f'\n\nPID: {self.accelerator.process_index}: \
                        The best `ckpt` file has saved in {self.best_ckpt_home}')
                self.accelerator.end_training()
                break
            elif early_stopper.counter == 0:
                self.save_ckpt(mode='best')

    @torch.no_grad()
    def valid_model_performance(self, **kwargs):
        pred_list, label_list = self.inference(*kwargs)
        pred = torch.hstack(pred_list).cpu()
        target = torch.hstack(label_list).cpu()
        f1_value = f1_score(y_true=target, y_pred=pred)
        acc_value = accuracy_score(y_true=target, y_pred=pred)
        recall_value = recall_score(y_true=target, y_pred=pred)
        pre_value = precision_score(y_true=target, y_pred=pred)
        print(acc_value,pre_value,recall_value,f1_value)


    @torch.no_grad()
    def inference(self, **kwargs):
        self.load_ckpt(mode='best')
        self.model = self.accelerator.prepare_model(self.model)
        self.test_loader = self.accelerator.prepare_data_loader(self.test_loader)

        self.model.eval()

        batch_iterator = tqdm(self.test_loader, desc=f'Pid: {self.accelerator.process_index}')

        pred_list = []
        label_list = []
        for idx, sample in enumerate(batch_iterator):
            data, label = sample
            logist = self.model(data)
            pred = F.softmax(logist, dim=-1).argmax(-1)
            pred_list.append(pred)
            label_list.append(label)
        pred_gathered = self.accelerator.gather(pred_list)
        label_gathered = self.accelerator.gather(label_list)

        return pred_gathered, label_gathered
