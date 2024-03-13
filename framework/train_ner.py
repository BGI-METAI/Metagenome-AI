#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/13/24 11:40 AM
# @Author  : zhangchao
# @File    : train_ner.py
# @Email   : zhangchao5@genomics.cn
import os.path as osp
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from typing import Optional, Union

import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from framework.classifier.loss_fn import ProteinLoss
from framework.dataset import CustomNERDataset
from framework.prottrans import ProtTransEmbeddings, PROTTRANS_T5_TYPE, PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE, \
    PROTTRANS_XLENT_TYPE, POOLING_ALL_TYPE, POOLING_CLS_TYPE, POOLING_MEAN_TYPE, POOLING_SUM_TYPE
from framework.train import ProteinAnnTrainer, TRAIN_LOADER_TYPE, TEST_LOADER_TYPE
from framework.utils import EarlyStopper


class ProteinNERTrainer(ProteinAnnTrainer):
    def __init__(
            self,
            pretrained_embedding_model_name_or_path: str,
            embedding_mode: Optional[
                Union[PROTTRANS_T5_TYPE, PROTTRANS_BERT_TYPE, PROTTRANS_ALBERT_TYPE, PROTTRANS_XLENT_TYPE]],
            **kwargs

    ):
        self.embedding_model = ProtTransEmbeddings(
            model_name_or_path=pretrained_embedding_model_name_or_path,
            mode=embedding_mode,
            **kwargs
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
        dataset = CustomNERDataset(processed_sequence_label_pairs_path=data_files)
        if mode == TRAIN_LOADER_TYPE:
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                **kwargs)
        if mode == TEST_LOADER_TYPE:
            self.test_loader = self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                collate_fn=self.collate_fn,
                shuffle=False,
                **kwargs)

    @torch.no_grad()
    def collate_fn(
            self,
            batch_sample,
            *,
            add_separator: bool = True,
            pooling: Optional[
                Union[POOLING_CLS_TYPE, POOLING_MEAN_TYPE, POOLING_SUM_TYPE, POOLING_ALL_TYPE]] = POOLING_ALL_TYPE,
    ):
        batch_seq, batch_label = [], []
        for sample in batch_sample:
            batch_seq.append(sample['seq'])
            batch_label.append(sample['label'])

        embed_vec = self.embedding_model.get_embedding(
            protein_seq=batch_seq,
            add_separator=add_separator,
            pooling=pooling
        )
        embed_vec = torch.tensor(embed_vec, device=self.device)

        for idx, val in enumerate(batch_label):
            tag_tensor = torch.tensor(val, device=self.device)
            batch_label[idx] = F.pad(tag_tensor, (0, embed_vec.shape[1] - tag_tensor.shape[0]))
        batch_label = torch.stack(batch_label)
        return embed_vec, batch_label

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
            project='ProteinAnnotationNER',
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
            batch_iterator = tqdm(self.train_loader, desc=f'Processing epoch: {eph:03d}')
            for batch in batch_iterator:
                x_data, y_tag = batch
                with torch.cuda.amp.autocast():
                    predict = self.model(x_data)
                    loss = ProteinLoss.cross_entropy_loss(
                        pred=predict.permute(0, 2, 1),
                        target=y_tag.long(),
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
