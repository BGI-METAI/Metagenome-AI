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

    @staticmethod
    def ddp_register(local_rank):
        """
        Initialize the PyTorch distributed backend. This is essential even for single machine, multiple GPU training
        dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0).
        The distributed process group contains all the processes that can communicate and synchronize with each other.
        """
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            # init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
            # rank=local_rank,
            # world_size=dist.get_world_size(),
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
        """each step to train the model."""

    @abstractmethod
    def valid_step(self, **kwargs):
        """validate the model every step of the way."""

    def train(self, config):
        self.optimizer = torch.optim.AdamW(
            self.classifier_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.99)
        scaler = torch.cuda.amp.GradScaler()

        early_stopper = EarlyStopper(patience=config.patience)

        self.wandb_register(
            config.user_name,
            project='proteinNER',
            model_folder=config.output_home
        )

        wandb.config({
            'learning_rate': config.learning_rate,
            'max_epoch': config.max_epoch,
            'batch_size': self.batch_size,
            'loss_weight': config.loss_weight})

        for eph in range(config.max_epoch):
            eph_loss = []
            self.classifier_model.train()
            self.train_loader.sampler.set_epoch(eph)
            batch_iterator = tqdm(self.train_loader, desc=f'Eph: {eph:03d}')
            for sample in batch_iterator:
                loss = self.train_step(sample=sample, loss_weight=config.loss_weight)

                # print(f' device: {dist.get_rank()} '.center(100, '*'))
                # print(f'loss: {loss.item()}')
                # print(f''.center(100, '*'))


                # if dist.get_rank() == 0:
                #     dist.barrier()
                dist.barrier()
                gather_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_loss, loss)
                gathered_mean_loss = torch.stack(gather_loss).mean().item()
                batch_iterator.set_postfix({'Loss': f'{gathered_mean_loss:.4f}'})
                eph_loss.append(gathered_mean_loss)
                wandb.log({'loss': gathered_mean_loss})

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            self.scheduler.step()

            with torch.no_grad():
                self.classifier_model.eval()
                all_accuracy = []
                for sample in self.test_loader:
                    accuracy = self.valid_step(sample=sample)

                    # if dist.get_rank() == 0:
                    #     dist.barrier()
                    dist.barrier()
                    gathered_acc = [torch.zeros_like(accuracy) for _ in range(dist.get_world_size())]
                    dist.all_gather(gathered_acc, accuracy)
                    gathered_mean_acc = torch.stack(gathered_acc).mean().item()
                    all_accuracy.append(gathered_mean_acc)

                wandb.log({'validation accuracy': np.mean(all_accuracy)})

            if early_stopper(np.mean(eph_loss)):
                print(f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')} Model Training Finished!")
                print(f'`ckpt` file has saved in {self.ckpt_home}')
                if config.load_best_model:
                    self.load_ckpt(ckpt_home=self.ckpt_home, reuse=False)
                break
            elif early_stopper.counter == 0:
                self.save_ckpt(ckpt_home=self.ckpt_home)


    def save_ckpt(self, ckpt_home):
        if dist.get_rank() == 0:
            save_dict = {
                'state_dict': self.classifier_model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'lr_schedule': self.scheduler.state_dict()
            }
            torch.save(save_dict,
                       os.path.join(ckpt_home, f'protein_ann_{self.classifier_model.module.__class__.__name__}.pth'))

    def load_ckpt(self, ckpt_home, reuse=False):
        checkpoint = torch.load(
            os.path.join(ckpt_home, f'protein_ann_{self.classifier_model.module.__class__.__name__}.pth'))
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
