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
import scipy.signal as signal
from scipy.stats import mode
from torch.nn.utils.rnn import pad_sequence

from proteinNER.base_module import BaseTrainer
from proteinNER.base_module.utils import EarlyStopper
from proteinNER.classifier.loss_fn import ProteinLoss
from proteinNER.classifier.evaluation_metrics import Eval_metrics


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
        length_threshold = kwargs.get('inference_length_threshold', 9)

        self.load_ckpt(mode='best')
        self.model, self.test_loader = self.accelerator.prepare(self.model, self.test_loader)
        self.model.eval()
        accuracy = []
        accuracy_exact_match = []
        precision = []
        label_list=[]
        pred_list=[]

        batch_iterator = tqdm(self.test_loader, desc=f'Pid: {self.accelerator.process_index}')
        for idx, sample in enumerate(batch_iterator):
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids, attention_mask)
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)
            pred_med = self.med_filter(pred, kernel_size=length_threshold) # length_threshold

            batch_padded_label, batch_padded_pred = self.pad_label(pred_med, batch_label,length_threshold) # pred_med
            pred_list.append(batch_padded_pred)
            label_list.append(batch_padded_label)

            # acc_sum = self.accuracy_exact_match(pred, batch_label, length_threshold) # pred_med
            # accuracy_exact_match.append(acc_sum.unsqueeze(dim=0))

            acc = self.accuracy_token_level(predict=pred, label=batch_label)
            accuracy.append(acc.unsqueeze(dim=0))

            # pres = self.precision_entity_level(predict=pred, label=batch_label)
            # precision.append(pres.unsqueeze(dim=0))

        # accuracy_exact_match = self.accelerator.gather(torch.cat(accuracy_exact_match, dim=0))
        accuracy = self.accelerator.gather(torch.cat(accuracy, dim=0))
        # precision = self.accelerator.gather(torch.cat(precision, dim=0))

        labels = self.accelerator.pad_across_processes(label_list, dim=1, pad_index=0)
        preds = self.accelerator.pad_across_processes(pred_list, dim=1, pad_index=0)
        labels_gathered = self.accelerator.gather(labels)
        preds_gathered = self.accelerator.gather(preds)
        # print(labels_gathered)
        # print(preds_gathered)

        eval_metrics = Eval_metrics()
        pre, rec, f1, acc = eval_metrics.evaluation(preds_gathered, labels_gathered)
        print(f"metrics result: precision={pre:.6f}, recall={rec:.6f}, f1-score={f1:.6f}, accuracy={acc:.6f}")

        self.accelerator.log({'Accuracy (Token Level)': accuracy.mean().item()})
        # self.accelerator.log({'Precision (Entity Level)': precision.mean().item()})
        # self.accelerator.log({'Accuracy (exact match)': (accuracy_exact_match.sum() / (self.test_loader.__len__()*self.batch_size*4)).item()})
        print('Accuracy (Token Level)', accuracy.mean().item())
        # print('Accuracy (exact match)', (accuracy_exact_match.sum() / (self.test_loader.__len__()*self.batch_size*4)).item())
        # print('precision', precision.mean().item())
        pass

    def accuracy_exact_match(self, predict, label, length_threshold):
        _, batch_pred_label = self.get_batch_pred_location_label(predict, length_threshold)
        _, batch_true_label = self.get_batch_pred_location_label(label, length_threshold)
        acc_total = 0
        for i in range(len(batch_pred_label)):
            acc_single = 0
            pred = batch_pred_label[i]
            target = batch_true_label[i]
            if not np.isnan(pred).any():
                pred = list(set(pred))
                target = list(set(target))
                if len(pred) == len(target):
                    for j in pred:
                        if j in target:
                            acc_single += 1
            if acc_single == len(target):
                acc_total += 1
        return torch.tensor(acc_total).to(self.accelerator.device)

    def pad_label(self, predict, label, length_threshold):
        _, batch_pred_label = self.get_batch_pred_location_label(predict, length_threshold)
        _, batch_true_label = self.get_batch_pred_location_label(label, length_threshold=1)

        batch_true_label = list(map(lambda x: torch.tensor(x), batch_true_label))
        batch_pred_label = list(map(lambda x: torch.tensor(x), batch_pred_label))
        padded_label = pad_sequence(batch_true_label, batch_first=True, padding_value=0).cuda()
        padded_pred = pad_sequence(batch_pred_label, batch_first=True, padding_value=0).cuda()

        return padded_label, padded_pred


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
        label_dict_path = kwargs.get('label_dict_path', '.')
        output_home = kwargs.get('output_home', '.')
        length_threshold = kwargs.get('inference_length_threshold', 9)

        self.load_ckpt(mode='best')
        self.model, self.test_loader = self.accelerator.prepare(self.model, self.test_loader)
        self.model.eval()

        file_name = 0
        batch_iterator = tqdm(self.test_loader, desc=f'Pid: {self.accelerator.process_index}')
        for idx, sample in enumerate(batch_iterator):
            t1 = time.time()
            file_name += 1
            input_ids, attention_mask, batch_label = sample
            logist = self.model(input_ids, attention_mask)
            pred = torch.nn.functional.softmax(logist, dim=-1).argmax(-1)
            batch_location_list, batch_label_name_list = self.get_batch_pred_location_label(pred, length_threshold,
                                                                                            label_dict_path)

            with open(output_home + '/result/inference_protein_location_label_batch' + str(file_name) + '.txt',
                      'w') as f:
                f.write('Sequence' + '\t' + 'Location([start, end])' + '\t' + 'Predicted_label' + '\n')
                for j in range(len(batch_label_name_list)):
                    f.write(
                        'ABC...' + '\t' + str(batch_location_list[j]) + '\t' + str(batch_label_name_list[j]) + '\n')
            t2 = time.time()
            print("each batch time", round(t2 - t1, 5), 'seconds')

    @staticmethod
    def convert_label(label_dict_path, label_id):
        with open(label_dict_path, "rb") as f:
            label_dict = pickle.load(f)
        label_name = label_dict[label_id.item()]
        return label_name

    def get_batch_pred_location_label(self, pred, length_threshold, label_dict_path=None):
        batch_location_list = []
        batch_label_name_list = []
        index_list = torch.nonzero(pred != 0, as_tuple=False)

        if index_list.size()[0] != 0:
            nonzero_label = pred[index_list[:, 0], index_list[:, 1]]

            diff_indices = torch.nonzero(nonzero_label[1:] != nonzero_label[:-1]).squeeze(dim=1)
            diff_indices = torch.cat([diff_indices, torch.tensor([len(nonzero_label) - 1]).cuda()])

            location_list = []
            label_name_list = []

            # diff_mask = index_list[:, 1][1:] - index_list[:, 1][:-1]  # 蛋白质分割（看gap的大小和负值情况）
            # single_indices = torch.nonzero(diff_mask < 0, as_tuple=True)[0]
            single_indices = []
            no_info_indices = [] # 指的是pred某个蛋白全预测0，需要记录一下，否则输出的pred维度不为3
            for i in range(self.batch_size):
                if torch.where(index_list[:, 0] == i)[0].numel() != 0:
                    single_indices.append(torch.max(torch.where(index_list[:, 0] == i)[0]).item())
                else:
                    no_info_indices.append(i)

            diff_indices = torch.cat((diff_indices, torch.tensor(single_indices).cuda())).unique().sort()[0]
            diff_indices = torch.cat([torch.tensor([0]).cuda(), diff_indices])

            for i in range(len(diff_indices)):
                # if diff_indices[i] in single_indices:
                #     batch_location_list.append(location_list) if len(location_list) != 0 else None
                #     batch_label_name_list.append(label_name_list) if len(label_name_list) != 0 else None
                #     location_list = []
                #     label_name_list = []

                if i > 0 and diff_indices[i] in single_indices:
                    batch_location_list.append(location_list if len(location_list) != 0 else [0])
                    batch_label_name_list.append(label_name_list if len(label_name_list) != 0 else [0])
                    location_list = []
                    label_name_list = []

                if i == 0:
                    start_position = index_list[:, 1][diff_indices[i]].item()
                    end_position = index_list[:, 1][diff_indices[i + 1]].item()
                    if (end_position - start_position) + 1 >= length_threshold:
                        location_list.append([start_position, end_position])
                        if label_dict_path != None:
                            label_name = self.convert_label(label_dict_path, nonzero_label[diff_indices[i] + 1])
                        else:
                            label_name = nonzero_label[diff_indices[i] + 1].item()
                        label_name_list.append(label_name)
                elif i < len(diff_indices) - 1:
                    start_position = index_list[:, 1][diff_indices[i] + 1].item()
                    end_position = index_list[:, 1][diff_indices[i + 1]].item()
                    if (end_position - start_position) + 1 >= length_threshold:  # [1,3]位置为1，2，3.因此长度为3-1+1
                        location_list.append([start_position, end_position])
                        if label_dict_path != None:
                            label_name = self.convert_label(label_dict_path, nonzero_label[diff_indices[i] + 1])
                        else:
                            label_name = nonzero_label[diff_indices[i] + 1].item()
                        label_name_list.append(label_name)
                else:
                    batch_location_list.append(location_list) if len(location_list) != 0 else None
                    batch_label_name_list.append(label_name_list) if len(label_name_list) != 0 else None
            if len(batch_label_name_list) != self.batch_size:
                for i in no_info_indices:
                    # batch_label_name_list.insert(i, float('nan'))
                    # batch_location_list.insert(i, float('nan'))
                    batch_label_name_list.insert(i, [0])
                    batch_location_list.insert(i, [0])
        else:
            for i in range(self.batch_size):
                batch_location_list.append([0])
                batch_label_name_list.append([0])

        return batch_location_list, batch_label_name_list



    @staticmethod
    def med_filter(pred, kernel_size):
        filtered_pred = np.zeros_like(pred.cpu())
        for i in range(pred.shape[0]):
            # pad_width = kernel_size // 2
            # pred = np.pad(pred, pad_width, mode='edge')

            for j in range(pred.shape[1]):
                window = pred[i][j:j + kernel_size]
                filtered_pred[i][j] = mode(window.cpu()).mode[0]

        return torch.tensor(filtered_pred).cuda()

    # def med_filter(pred, kernel_size):
    #     pred_list = []
    #     for i in range(pred.shape[0]):
    #         pred_med = signal.medfilt(pred[i].cpu(), kernel_size=kernel_size)
    #         pred_list.append(pred_med)
    #     return torch.tensor(pred_list).cuda()




