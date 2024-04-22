#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:12
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import torch
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
        self.loss_weight = kwargs.get('loss_weight', 1.)

        early_stopper = EarlyStopper(patience=kwargs.get('patience', 4))
        self.register_wandb(
            user_name=kwargs.get('username', 'zhangchao162'),
            project_name=kwargs.get('project', 'ProtT5'),
            group=kwargs.get('group', 'training')
        )

        self.model, self.optimizer, self.train_loader, self.test_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.test_loader, self.lr_scheduler
        )

        for eph in range(kwargs.get('epoch', 100)):
            self.model.train()
            batch_iterator = tqdm(self.train_loader,
                                  desc=f'Pid: {self.accelerator.process_index} Eph: {eph:03d} ({early_stopper.counter} / {early_stopper.patience})')
            eph_loss = []
            for idx, sample in enumerate(batch_iterator):
                input_ids, attention_mask, batch_label = sample
                with self.accelerator.accumulate(self.model):
                    logist = self.model(input_ids, attention_mask)
                    with self.accelerator.autocast():
                        loss = ProteinLoss.focal_loss(
                            pred=logist.permute(0, 2, 1),
                            target=batch_label,
                            weight=self.loss_weight,
                            gamma=2.
                        )

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                self.accelerator.log({'loss': loss.item()})
                eph_loss.append(loss.item())
                if self.save_in_batch and idx % 500 == 0:
                    self.save_ckpt('batch')

                with self.accelerator.main_process_first():
                    self.accelerator.log({'learning rate': self.optimizer.state_dict()['param_groups'][0]['lr']})
            self.lr_scheduler.step()

            # if eph % 5 == 0:
            #     self.valid_model_performance(test_loader=self.test_loader)

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
    def valid_model_performance(self, test_loader):
        self.model.eval()
        accuracy = []
        precision = []
        batch_iterator = tqdm(test_loader, desc=f'Pid: {self.accelerator.process_index}')
        for idx, sample in enumerate(batch_iterator):
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids, attention_mask)
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)

            acc = self.accuracy_token_level(predict=pred, label=batch_label)
            accuracy.append(acc.unsqueeze(dim=0))

            pres = self.precision_entity_level(predict=pred, label=batch_label)
            precision.append(pres.unsqueeze(dim=0))

        accuracy = self.accelerator.gather(torch.cat(accuracy, dim=0))
        precision = self.accelerator.gather(torch.cat(precision, dim=0))

        self.accelerator.log({'Accuracy (Token Level)': accuracy.mean().item()})
        self.accelerator.log({'Precision (Entity Level)': np.mean(precision)})

    @staticmethod
    def accuracy_token_level(predict, label):
        return torch.eq(predict, label).float().mean()

    def precision_entity_level(self, predict, label):
        correct_dict = {i: torch.zeros(1).to(self.accelerator.device) for i in
                        range(1, self.model.classifier.out_features)}
        predict_dict = {i: torch.zeros(1).to(self.accelerator.device) for i in
                        range(1, self.model.classifier.out_features)}
        label_dict = {i: torch.zeros(1).to(self.accelerator.device) for i in
                      range(1, self.model.classifier.out_features)}

        predict = predict.flatten()
        label = label.flatten()
        mask = label.bool()

        correct_tag = (torch.eq(predict, label) * mask).float()
        correct_start_pos, correct_end_pos = self.get_position_interval_of_consecutive_nonzero_values(correct_tag)
        if correct_start_pos:
            for pos in correct_start_pos:
                correct_dict[label[pos].item()] += 1

        predict_start_pos, predict_end_pos = self.get_position_interval_of_consecutive_nonzero_values(predict)
        if predict_start_pos:
            for pos in predict_start_pos:
                predict_dict[predict[pos].item()] += 1

        label_start_pos, label_end_pos = self.get_position_interval_of_consecutive_nonzero_values(label)
        if label_start_pos:
            for pos in label_start_pos:
                label_dict[label[pos].item()] += 1

        precision_list = []
        for k in correct_dict.keys():
            if predict_dict[k] == 0 and label_dict[k] == 0:
                continue

            if predict_dict[k] == 0:
                score = torch.zeros(1).to(self.accelerator.device)
            else:
                score = correct_dict[k] / predict_dict[k]
            precision_list.append(score)
        return torch.tensor(precision_list).to(self.accelerator.device).float().mean()

    @staticmethod
    def get_position_interval_of_consecutive_nonzero_values(data):
        data = data.flatten()
        nonzero_indices = torch.nonzero(data)
        nonzero_values = data[nonzero_indices]
        start_positions = []
        end_positions = []

        current_start = None
        prev_value = None
        for i in range(len(nonzero_indices)):
            value = nonzero_values[i]
            if current_start is None:
                current_start = nonzero_indices[i]
                prev_value = value
            elif value != prev_value:
                start_positions.append(current_start.item())
                end_positions.append(nonzero_indices[i - 1].item())
                current_start = nonzero_indices[i]
                prev_value = value

        if current_start is not None:
            start_positions.append(current_start.item())
            end_positions.append(nonzero_indices[-1].item())

        return start_positions, end_positions

    @torch.no_grad()
    def inference(self, **kwargs):
        pass
