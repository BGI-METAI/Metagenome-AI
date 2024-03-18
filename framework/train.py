#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/13/24 11:40 AM
# @Author  : zhangchao
# @File    : train.py
# @Email   : zhangchao5@genomics.cn
import os
import wandb
import torch
import torch.distributed as dist

import os.path as osp
from datetime import datetime
from pathlib import Path
from typing import Optional, Union
from torch.utils.data import DataLoader
from tqdm import tqdm

from framework.classifier.loss_fn import ProteinLoss
from framework.dataset import CustomNERDataset, SequentialDistributedSampler
from framework.prottrans import ProtTransEmbeddings
from framework.base_train import ProteinAnnBaseTrainer, TRAIN_LOADER_TYPE, TEST_LOADER_TYPE
from framework.utils import EarlyStopper


class ProteinNERTrainer(ProteinAnnBaseTrainer):
    def __init__(self, config, **kwargs):

        # initialize DDP
        self.ddp_register(rank=config.local_rank, world_size=config.world_size)

        self.embedding_model = ProtTransEmbeddings(
            model_name_or_path=config.model_path_or_name,
            mode=config.embed_mode,
            local_rank=config.local_rank,
        )

    def dataset_register(
            self,
            data_files,
            *,
            batch_size: int = 1024,
            mode: Optional[Union[TRAIN_LOADER_TYPE, TEST_LOADER_TYPE]] = TRAIN_LOADER_TYPE,
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

        if mode == TRAIN_LOADER_TYPE:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                # num_replicas=dist.get_world_size(),
                # rank=dist.get_rank(),
                # shuffle=False
            )
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=train_sampler)
        if mode == TEST_LOADER_TYPE:
            test_sampler = SequentialDistributedSampler(dataset=dataset, batch_size=batch_size)
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                sampler=test_sampler)

    def train(
            self,
            max_epoch=100,
            learning_rate=1e-6,
            weight_decay=5e-4,
            patience=10,
            *,
            user_name='zhangchao162',
            load_best_model=True,
            **kwargs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()

        loss_weight = kwargs.get('loss_weight', 1.)
        output_home = kwargs.get('output_home', './protein_anno_NER')
        ckpt_home = osp.join(output_home, 'ckpt')
        Path(ckpt_home).mkdir(parents=True, exist_ok=True)

        self.wandb_register(
            user_name,
            project='proteinNER',
            model_folder=output_home
        )

        wandb.config = {
            'learning_rate': learning_rate,
            'max_epoch': max_epoch,
            'batch_size': self.batch_size,
            'loss_weight': loss_weight}

        early_stopper = EarlyStopper(patience=patience)

        for eph in range(max_epoch):
            eph_loss = 0
            self.train_loader.sampler.set_epoch(eph)
            batch_iterator = tqdm(self.train_loader, desc=f'Processing epoch: {eph:03d}')
            for sample in batch_iterator:
                input_ids, attention_mask, batch_label = sample
                with torch.no_grad():
                    x_data = self.embedding_model.get_embedding(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pooling='all'
                    )
                with torch.cuda.amp.autocast():
                    predict = self.model(x_data)
                    loss = ProteinLoss.cross_entropy_loss(
                        pred=predict.permute(0, 2, 1),
                        target=batch_label.long().to(self.device),
                        weight=loss_weight
                    )
                eph_loss += loss.item()
                batch_iterator.set_postfix({'Training loss': f'{loss.item():.4f}'})
                wandb.log({'loss': loss.item()})
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            scheduler.step()
            if early_stopper.counter == 0:
                self.save_ckpt(ckpt_home=ckpt_home)
            if early_stopper(eph_loss):
                print(f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")
                print(f'`ckpt` file has saved in {ckpt_home}')
                if load_best_model:
                    self.load_ckpt(ckpt_home=ckpt_home)
                break
