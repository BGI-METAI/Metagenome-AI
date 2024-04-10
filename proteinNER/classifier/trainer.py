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
        predicts = []
        labels = []
        for sample in self.test_loader:
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids.cuda(), attention_mask.cuda())
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)
            predicts.append(pred)
            labels.append(batch_label)
        predicts = self.distributed_concat(torch.concat(predicts, dim=0), len(self.test_loader))
        labels = self.distributed_concat(torch.concat(labels, dim=0), len(self.test_loader))
        accuracy = self.accuracy_token_level(predicts, labels)
        precision = self.precision_entity_level(predicts, labels)
        wandb.log({'Accuracy': np.mean(accuracy.item())})
        wandb.log({'Precision': np.mean(precision)})

    @staticmethod
    def accuracy_token_level(predict, label):
        return torch.eq(predict, label).float().mean()

    def precision_entity_level(self, predict, label):
        correct_dict = {i: 0 for i in range(1, label.max().item() + 1)}
        predict_dict = {i: 0 for i in range(1, label.max().item() + 1)}
        label_dict = {i: 0 for i in range(1, label.max().item() + 1)}

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
                predict_dict[label[pos].item()] += 1

        label_start_pos, label_end_pos = self.get_position_interval_of_consecutive_nonzero_values(label)
        if label_start_pos:
            for pos in label_start_pos:
                label_dict[label[pos].item()] += 1

        precision_list = []
        for k in correct_dict.keys():
            precision_list.append(correct_dict[k] / predict_dict[k])
        return precision_list

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
