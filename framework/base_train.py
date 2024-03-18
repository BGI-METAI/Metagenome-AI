#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:26 PM
# @Author  : zhangchao
# @File    : base_train.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import wandb
import socket
import torch
import torch.distributed as dist
import pandas as pd

from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from framework.classifier import FullLinearClassifier, AminoAcidsNERClassifier
from framework.classifier.loss_fn import ProteinLoss
from framework.dataset import CustomDataFrameDataset, CustomNERDataset
from framework.utils import EarlyStopper

TRAIN_LOADER_TYPE: str = 'train'
VALID_LOADER_TYPE: str = 'valid'
TEST_LOADER_TYPE: str = 'test'


class ProteinAnnBaseTrainer:
    model: Optional[Union[FullLinearClassifier, AminoAcidsNERClassifier]] = None
    device: torch.device = None
    train_loader: Optional[Union[DataLoader, CustomNERDataset]] = None
    valid_loader: Optional[Union[DataLoader, CustomNERDataset]] = None
    test_loader: Optional[Union[DataLoader, CustomNERDataset]] = None
    batch_size: int = 1

    def wandb_register(
            self,
            user_name: str,
            project: str = 'protein function annotation',
            group: str = 'classifier',
            model_folder: str = '.',
            timestamp: str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ):
        """
        register wandb to ProteintTrainer

        :param user_name:
            username or team name where you're sending runs.
            This entity must exist before you can send runs there,
            so make sure to create your account or team in the UI before starting to log runs.
            If you don't specify an entity, the run will be sent to your default entity,
            which is usually your username. Change your default entity in [your settings](https://wandb.ai/settings)
            under "default location to create new projects".
        :param project:
            The name of the project where you're sending the new run. If the project is not specified, the run is put in an "Uncategorized" project.
        :param group:
            Specify a group to organize individual runs into a larger experiment.
            For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate
            a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this
            on and off in the UI. For more details, see our [guide to grouping runs](https://docs.wandb.com/guides/runs/grouping).
        :param model_folder:
            wandb ouput file save path
        :param timestamp:
        :return:
        """
        wandb_output_dir = osp.join(model_folder, 'wandb_home')
        Path(wandb_output_dir).mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=project,
            entity=user_name,
            notes=socket.gethostname(),
            name=f'prot_func_anno_{timestamp}',
            group=group,
            dir=wandb_output_dir,
            job_type='training',
            reinit=True
        )
        wandb.watch(self.model, log='all')

    @abstractmethod
    def dataset_register(
            self,
            protein_dataset: pd.DataFrame,
            *,
            mode: Optional[Union[TRAIN_LOADER_TYPE, VALID_LOADER_TYPE, TEST_LOADER_TYPE]] = 'train',
            batch_size: int = 1024,
            **kwargs
    ):
        """register dataset"""

    def ddp_register(self, rank, world_size):
        """
        Initialize the PyTorch distributed backend. This is essential even for single machine, multiple GPU training
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0).
        The distributed process group contains all the processes that can communicate and synchronize with each other.

        :param rank:
            Unique identifier of each process
        :param world_size:
            Total number of processes
        :return:
        """
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "12356"

        torch.cuda.set_device(rank)
        dist.init_process_group(
            backend="nccl",
            # init_method='tcp://172.16.7.39:12356',
            rank=rank,
            world_size=world_size,
            # timeout=timedelta(seconds=10)
        )

    def model_register(
            self, model: Optional[Union[FullLinearClassifier, AminoAcidsNERClassifier]],
            **kwargs
    ):
        """
        register protein sequence model to ProteintTrainer

        :param model:
            protein sequence model
        :return:
        """

        local_rank = kwargs.get('local_rank')
        device = torch.device('cuda', local_rank)
        self.model = DDP(model.to(device), device_ids=[local_rank])

    def train(
            self,
            max_epoch,
            learning_rate,
            weight_decay=5e-4,
            patience=10,
            *,
            user_name,
            load_best_model=True,
            **kwargs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        scaler = torch.cuda.amp.GradScaler()
        self.model.train()

        early_stopper = EarlyStopper(patience=patience)

        loss_weight = kwargs.get('loss_weight', default=1.)
        output_home = kwargs.get('output_home', default='.')
        ckpt_home = osp.join(output_home, 'ckpt')
        Path(ckpt_home).mkdir(parents=True, exist_ok=True)

        self.wandb_register(user_name, model_folder=output_home)

        wandb.config({
            'learning_rate': learning_rate,
            'max_epoch': max_epoch,
            'batch_size': self.batch_size,
            'loss_weight': loss_weight})

        for eph in range(max_epoch):
            eph_loss = 0
            batch_iterator = tqdm(self.train_loader, desc=f'Processing epoch: {eph:03d}')
            for batch in batch_iterator:
                x_data, y_target = batch
                x_data = x_data.to(self.device)
                y_target = y_target.to(self.device)

                with torch.cuda.amp.autocast():
                    predict = self.model(x_data)
                    loss = ProteinLoss.cross_entropy_loss(pred=predict, target=y_target, weight=loss_weight)
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

    def save_ckpt(self, ckpt_home):
        if dist.get_rank() == 0:
            torch.save(self.model.module.state_dict(),
                       os.path.join(ckpt_home, f'protein_ann_{self.model.__name__}.bgi'))

    def load_ckpt(self, ckpt_home):
        checkpoint = torch.load(
            os.path.join(ckpt_home, f'protein_ann_{self.model.__name__}.bgi'),
            map_location=lambda storage, loc: storage)
        state_dict = self.model.module.state_dict()
        trained_dict = {k: v for k, v in checkpoint.items() if k in state_dict}
        state_dict.update(trained_dict)
        self.model.module.load_state_dict(state_dict)
