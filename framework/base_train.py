#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/8/24 4:26 PM
# @Author  : zhangchao
# @File    : base_train.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
# import wandb
import socket
import numpy as np
import random
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import pandas as pd

from sklearn import metrics
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from framework import CLS, EMB
from framework.classifier import FullLinearClassifier, AminoAcidsNERClassifier
from framework.dataset import CustomNERDataset
from framework.prottrans import ProtTransEmbeddings
from framework.utils import EarlyStopper
from framework.embeddings import Embeddings

TRAIN_LOADER_TYPE: str = 'train'
VALID_LOADER_TYPE: str = 'valid'
TEST_LOADER_TYPE: str = 'test'


class ProteinAnnBaseTrainer(ABC):
    embedding_model: Embeddings = None
    classifier_model: Optional[Union[FullLinearClassifier, AminoAcidsNERClassifier]] = None
    optimizer: Optimizer = None
    scheduler: LRScheduler = None
    train_loader: Optional[Union[DataLoader, CustomNERDataset]] = None
    valid_loader: Optional[Union[DataLoader, CustomNERDataset]] = None
    test_loader: Optional[Union[DataLoader, CustomNERDataset]] = None
    batch_size: int = 1

    def __init__(self, config, **kwargs):
        self.ckpt_home = osp.join(config.output_home, 'ckpt')
        Path(self.ckpt_home).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def init_seeds(seed, cuda_deterministic=True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda_deterministic:  # slower, more reproducible
            cudnn.deterministic = True
            cudnn.benchmark = False
        else:  # faster, less reproducible
            cudnn.deterministic = False
            cudnn.benchmark = True

    # def wandb_register(
    #         self,
    #         user_name: str,
    #         project: str = 'protein function annotation',
    #         group: str = 'classifier',
    #         model_folder: str = '.',
    #         timestamp: str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    # ):
    #     """
    #     register wandb to ProteintTrainer
    #
    #     :param user_name:
    #         username or team name where you're sending runs.
    #         This entity must exist before you can send runs there,
    #         so make sure to create your account or team in the UI before starting to log runs.
    #         If you don't specify an entity, the run will be sent to your default entity,
    #         which is usually your username. Change your default entity in [your settings](https://wandb.ai/settings)
    #         under "default location to create new projects".
    #     :param project:
    #         The name of the project where you're sending the new run. If the project is not specified, the run is put in an "Uncategorized" project.
    #     :param group:
    #         Specify a group to organize individual runs into a larger experiment.
    #         For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate
    #         a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this
    #         on and off in the UI. For more details, see our [guide to grouping runs](https://docs.wandb.com/guides/runs/grouping).
    #     :param model_folder:
    #         wandb ouput file save path
    #     :param timestamp:
    #     :return:
    #     """
    #     wandb_output_dir = osp.join(model_folder, 'wandb_home')
    #     Path(wandb_output_dir).mkdir(parents=True, exist_ok=True)
    #     wandb.init(
    #         project=project,
    #         entity=user_name,
    #         notes=socket.gethostname(),
    #         name=f'prot_func_anno_{timestamp}',
    #         group=group,
    #         dir=wandb_output_dir,
    #         job_type='training',
    #         reinit=True
    #     )
    #     wandb.watch(self.model, log='all')

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

    def ddp_register(self, local_rank):
        """
        Initialize the PyTorch distributed backend. This is essential even for single machine, multiple GPU training
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0).
        The distributed process group contains all the processes that can communicate and synchronize with each other.
        """
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=30)
        )

    @staticmethod
    def ddp_clean():
        dist.destroy_process_group()

    def model_register(
            self,
            model: Optional[Union[FullLinearClassifier, AminoAcidsNERClassifier, ProtTransEmbeddings]],
            model_type: Optional[Union[CLS, EMB]],
            **kwargs
    ):
        """
        register protein sequence model to ProteintTrainer

        :param model:
            protein sequence model
        :param model_type:
            model category
        :return:
        """
        local_rank = kwargs.get('local_rank')
        reuse = kwargs.get('reuse')

        if model_type == EMB:
            self.embedding_model = model
            self.embedding_model.cuda()
            # self.embedding_model.model = DDP(self.embedding_model.model, device_ids=[local_rank], output_device=local_rank)
        elif model_type == CLS:
            if not reuse:
                self.classifier_model = model.cuda()
            else:
                self.load_ckpt(ckpt_home=self.ckpt_home, reuse=True)
            self.classifier_model = DDP(self.classifier_model, device_ids=[local_rank], output_device=local_rank)
        else:
            raise ValueError('Got an invalid model category, only support `CLS` and `EMB`!')

    @abstractmethod
    def train_step(self, **kwargs):
        """"""

    def train(self, config):
        self.optimizer = torch.optim.AdamW(
            self.classifier_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        scaler = torch.cuda.amp.GradScaler()
        self.classifier_model.train()

        early_stopper = EarlyStopper(patience=config.patience)

        # self.wandb_register(
        #     user_name,
        #     project='proteinNER',
        #     model_folder=output_home
        # )

        # wandb.config({
        #     'learning_rate': learning_rate,
        #     'max_epoch': max_epoch,
        #     'batch_size': self.batch_size,
        #     'loss_weight': loss_weight})

        for eph in range(config.max_epoch):
            eph_loss = 0
            self.train_loader.sampler.set_epoch(eph)
            batch_iterator = tqdm(self.train_loader, desc=f'Eph: {eph:03d}')
            for sample in batch_iterator:
                loss = self.train_step(sample=sample, loss_weight=config.loss_weight)
                eph_loss += loss.item()
                batch_iterator.set_postfix({'Loss': f'{loss.item():.4f}'})
                # wandb.log({'loss': loss.item()})
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            self.scheduler.step()
            if early_stopper.counter == 0:
                self.save_ckpt(ckpt_home=self.ckpt_home)
            if early_stopper(eph_loss):
                print(f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")
                print(f'`ckpt` file has saved in {self.ckpt_home}')
                if config.load_best_model:
                    self.load_ckpt(ckpt_home=self.ckpt_home, reuse=False)
                break

    def save_ckpt(self, ckpt_home):
        if dist.get_rank() == 0:
            save_dict = {
                'state_dict': self.classifier_model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.scheduler.state_dict()
            }
            torch.save(save_dict,
                       os.path.join(ckpt_home, f'protein_ann_{self.classifier_model.__name__}.bgi'))

    def load_ckpt(self, ckpt_home, reuse=False):
        checkpoint = torch.load(os.path.join(ckpt_home, f'protein_ann_{self.classifier_model.__name__}.bgi'))
        state_dict = self.classifier_model.module.state_dict()
        trained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in state_dict}
        state_dict.update(trained_dict)
        self.classifier_model.module.load_state_dict(state_dict)
        if reuse:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['lr_schedule'])

    @staticmethod
    def metric_score(inputs, label):
        accuracy = metrics.accuracy_score(label, inputs)
        precision = metrics.precision_score(label, inputs, average='macro')
        recall = metrics.recall_score(label, inputs, average='macro')
        f1 = metrics.f1_score(label, inputs, average='macro')
        return accuracy, precision, recall, f1

