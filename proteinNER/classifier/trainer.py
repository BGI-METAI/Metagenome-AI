#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 14:12
# @Author  : zhangchao
# @File    : trainer.py
# @Email   : zhangchao5@genomics.cn
import torch
import numpy as np
import pickle
import time

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
            user_name=kwargs.get('username', 'kxzhang2000'),
            project_name=kwargs.get('project', 'Pro_func'),
            group=kwargs.get('group', 'NER_V2')
        )

        self.model, self.optimizer, self.train_loader, self.test_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.test_loader, self.lr_scheduler
        )

        for eph in range(kwargs.get('epoch', 10)):  # 100
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

            self.valid_model_performance(test_loader=self.test_loader)

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
        for sample in test_loader:
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
        self.accelerator.log({'Precision (Entity Level)': precision.mean().item()})

    @staticmethod
    def accuracy_token_level(predict, label):
        return torch.eq(predict, label).float().mean()

    def precision_entity_level(self, predict, label):
        correct_dict = {i: torch.zeros(1).to(self.accelerator.device) for i in
                        range(1, self.model.module.classifier.out_features)}
        predict_dict = {i: torch.zeros(1).to(self.accelerator.device) for i in
                        range(1, self.model.module.classifier.out_features)}
        label_dict = {i: torch.zeros(1).to(self.accelerator.device) for i in
                      range(1, self.model.module.classifier.out_features)}

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
        label_dict_path = kwargs.get('label_dict_path', '.')
        output_home = kwargs.get('output_home', '.')
        length_threshold = kwargs.get('inference_length_threshold', 50)

        self.load_ckpt(mode='best')
        self.model.eval()

        file_name = 0
        for sample in self.test_loader:
            t1 = time.time()
            file_name += 1
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids.cuda(), attention_mask.cuda())
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)

            index_list = torch.nonzero(pred != 0, as_tuple=False)
            if index_list.size()[0] != 0:
                nonzero_label = pred[index_list[:, 0], index_list[:, 1]]

                diff_indices = torch.nonzero(nonzero_label[1:] != nonzero_label[:-1]).squeeze()
                diff_indices = torch.cat([diff_indices, torch.tensor([len(nonzero_label) - 1]).cuda()])

                diff_mask = index_list[:, 1][1:] - index_list[:, 1][:-1]  # 蛋白质分割（看gap的大小和负值情况）
                single_indices = torch.nonzero(diff_mask < 0, as_tuple=True)[0]
                diff_indices = torch.cat((diff_indices, single_indices)).unique().sort()[0]
                diff_indices = torch.cat([torch.tensor([0]).cuda(), diff_indices])

                batch_location_list = []
                batch_label_name_list = []
                location_list = []
                label_name_list = []

                for i in range(len(diff_indices)):
                    if diff_indices[i] in single_indices:
                        batch_location_list.append(location_list) if len(location_list) != 0 else None
                        batch_label_name_list.append(label_name_list) if len(label_name_list) != 0 else None
                        location_list = []
                        label_name_list = []

                    if i == 0:
                        start_position = index_list[:, 1][diff_indices[i]].item()
                        end_position = index_list[:, 1][diff_indices[i + 1]].item()
                        if (end_position - start_position) + 1 >= length_threshold:
                            location_list.append([start_position, end_position])
                            label_name = self.convert_label(label_dict_path, nonzero_label[diff_indices[i] + 1])
                            label_name_list.append(label_name)
                    elif i < len(diff_indices) - 1:
                        start_position = index_list[:, 1][diff_indices[i] + 1].item()
                        end_position = index_list[:, 1][diff_indices[i + 1]].item()
                        if (end_position - start_position) + 1 >= length_threshold:  # [1,3]位置为1，2，3.因此长度为3-1+1
                            location_list.append([start_position, end_position])
                            label_name = self.convert_label(label_dict_path, nonzero_label[diff_indices[i] + 1])
                            label_name_list.append(label_name)
                    else:
                        batch_location_list.append(location_list) if len(location_list) != 0 else None
                        batch_label_name_list.append(label_name_list) if len(label_name_list) != 0 else None
                with open(output_home + '/inference_protein_location_label_batch' + str(file_name) + '.txt', 'w') as f:
                    f.write('Sequence' + '\t' + 'Location([start, end])' + '\t' + 'Predicted_label' + '\n')
                    for j in range(len(batch_label_name_list)):
                        f.write(
                            'ABC...' + '\t' + str(batch_location_list[j]) + '\t' + str(batch_label_name_list[j]) + '\n')
            t2 = time.time()
            print("ench batch time", round(t2 - t1, 5), 'seconds')

    @staticmethod
    def convert_label(label_dict_path, label_id):
        with open(label_dict_path, "rb") as f:
            label_dict = pickle.load(f)
        label_name = label_dict[label_id.item()]
        return label_name
