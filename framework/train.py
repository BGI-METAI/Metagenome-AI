#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/13/24 11:40 AM
# @Author  : zhangchao
# @File    : train.py
# @Email   : zhangchao5@genomics.cn
import torch
from typing import Optional, Union
from torch.utils.data import DataLoader

from framework.classifier.loss_fn import ProteinLoss
from framework.dataset import CustomNERDataset, SequentialDistributedSampler
from framework.base_train import ProteinAnnBaseTrainer, TRAIN_LOADER_TYPE, TEST_LOADER_TYPE


class ProteinNERTrainer(ProteinAnnBaseTrainer):
    def dataset_register(
            self,
            data_files,
            *,
            batch_size: int = 1024,
            data_type: Optional[Union[TRAIN_LOADER_TYPE, TEST_LOADER_TYPE]] = TRAIN_LOADER_TYPE,
            **kwargs
    ):
        self.batch_size = batch_size
        tokenizer_model_name_or_path = kwargs.get('tokenizer_model_name_or_path')
        tokenizer_mode = kwargs.get('tokenizer_mode')
        legacy = kwargs.get('legacy')
        do_lower_case = kwargs.get('do_lower_case')

        dataset = CustomNERDataset(
            processed_sequence_label_pairs_path=data_files,
            tokenizer_model_name_or_path=tokenizer_model_name_or_path,
            mode=tokenizer_mode,
            legacy=legacy,
            do_lower_case=do_lower_case
        )

        if data_type == TRAIN_LOADER_TYPE:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=train_sampler)
        if data_type == TEST_LOADER_TYPE:
            test_sampler = SequentialDistributedSampler(dataset=dataset, batch_size=batch_size)
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=test_sampler)

    def train_step(self, sample, **kwargs):
        loss_weight = kwargs.get('loss_weight', 1.)
        input_ids, attention_mask, batch_label = sample

        with torch.no_grad():
            x_data = self.embedding_model.get_embedding(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pooling='all'
            )
        with torch.cuda.amp.autocast():
            predict = self.classifier_model(x_data)
            # loss = ProteinLoss.cross_entropy_loss(
            #     pred=predict.permute(0, 2, 1),
            #     target=batch_label,
            #     weight=loss_weight
            # )
            loss = ProteinLoss.focal_loss(
                pred=predict.permute(0, 2, 1),
                target=batch_label,
                weight=loss_weight,
                gamma=2.
            )

        del x_data, predict
        torch.cuda.empty_cache()

        return loss

