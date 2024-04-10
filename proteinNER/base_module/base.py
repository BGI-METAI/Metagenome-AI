#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 10:06
# @Author  : zhangchao
# @File    : base.py
# @Email   : zhangchao5@genomics.cn
import os
import os.path as osp
import socket
import random
import numpy as np
import wandb
import torch
import torch.distributed as dist

from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from datetime import timedelta, datetime
from typing import Optional, Union

from peft import PeftModel
from torch.backends import cudnn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from proteinNER.base_module import CustomNERDataset, SequentialDistributedSampler
from proteinNER.base_module.dataset import CustomPEFTEmbeddingDataset
from proteinNER.classifier.model import ProtTransT5ForAAClassifier


class BaseTrainer(ABC):
    model: Optional[Union[torch.nn.Module, ProtTransT5ForAAClassifier]] = field(default=None, metadata={
        "help": "protein sequence AA classifier model"})
    train_loader: DataLoader = field(default=None, metadata={"help": "train loader"})
    test_loader: DataLoader = field(default=None, metadata={"help": "test loader"})
    valid_loader: DataLoader = field(default=None, metadata={"help": "valid loader"})
    optimizer: Optimizer = field(default=None, metadata={"help": "optimizer for training model"})
    lr_scheduler: LRScheduler = field(default=None, metadata={"help": "Learning rate decay course schedule"})
    scaler: torch.cuda.amp.GradScaler = field(default=None, metadata={"help": "GradScaler"})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    learning_rate: float = field(default=1e-3, metadata={"help": "learning rate"})
    is_trainable: bool = field(default=True, metadata={"help": "whether the model to be train or not"})
    reuse: bool = field(default=False, metadata={"help": "whether the model parameters to be reuse or not"})

    def __init__(self, **kwargs):
        # output home
        output_home = kwargs.get('output_home', '.')
        self.save_in_batch = kwargs.get('save_in_batch', True)
        # ckpt
        self.best_ckpt_home = self.register_dir(output_home, 'best_ckpt')
        if self.save_in_batch:
            self.batch_ckpt_home = self.register_dir(output_home, 'batch_ckpt')
        # wandb
        self.wandb_home = self.register_dir(output_home, 'wandb')
        # result
        self.result_home = self.register_dir(output_home, 'result')
        # except
        self.except_home = self.register_dir(output_home, 'except')
        # register DDP
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(
            backend='nccl',
            timeout=timedelta(minutes=60)
        )
        rank = dist.get_rank()
        self.init_seeds(seed=rank + 1)

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

    @staticmethod
    def register_dir(parent_path, folder):
        new_path = osp.join(parent_path, folder)
        Path(new_path).mkdir(parents=True, exist_ok=True)
        return new_path

    def register_wandb(
            self,
            user_name: str,
            project_name: str = 'ProteinSequenceAAClassifier',
            group: str = 'NER',
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
        :param project_name:
            The name of the project where you're sending the new run. If the project is not specified, the run is put in an "Uncategorized" project.
        :param group:
            Specify a group to organize individual runs into a larger experiment.
            For example, you might be doing cross validation, or you might have multiple jobs that train and evaluate
            a model against different test sets. Group gives you a way to organize runs together into a larger whole, and you can toggle this
            on and off in the UI. For more details, see our [guide to grouping runs](https://docs.wandb.com/guides/runs/grouping).
            wandb ouput file save path
        :param timestamp:
        """
        wandb.init(
            project=project_name,
            entity=user_name,
            notes=socket.gethostname(),
            name=f'ProteinSeqNER_{timestamp}',
            group=group,
            dir=self.wandb_home,
            job_type='training',
            reinit=True
        )
        wandb.watch(self.model, log='all')

    def register_dataset(
            self,
            data_files,
            mode,
            dataset_type='class',
            **kwargs
    ):
        """

        :param data_files: pickle files path list of protein sequence
        :param mode: data loader type, optional, only support `train`, `test` and `valid`
        :param dataset_type: dataset type, optional, only support `class` and `embed`

        :return:
        """
        self.batch_size = kwargs.get('batch_size', 1)
        model_name_or_path = kwargs.get('model_name_or_path')
        legacy = kwargs.get('legacy', False)
        do_lower_case = kwargs.get('do_lower_case', False)

        if dataset_type == 'class':
            dataset = CustomNERDataset(
                processed_sequence_label_pairs_path=data_files,
                tokenizer_model_name_or_path=model_name_or_path,
                legacy=legacy,
                do_lower_case=do_lower_case
            )
        elif dataset_type == 'embed':
            dataset = CustomPEFTEmbeddingDataset(
                incremental_protein_sequence_path=data_files,
                tokenizer_model_name_or_path=model_name_or_path,
                legacy=legacy,
                do_lower_case=do_lower_case
            )
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `class` and `embed`')

        if mode == 'train':
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.collate_fn,
                sampler=train_sampler
            )
        elif mode == 'test':
            test_sampler = SequentialDistributedSampler(dataset=dataset, batch_size=self.batch_size)
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.collate_fn,
                sampler=test_sampler
            )
        elif mode == 'valid':
            valid_sampler = SequentialDistributedSampler(dataset=dataset, batch_size=self.batch_size)
            self.valid_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.collate_fn,
                sampler=valid_sampler
            )
        else:
            raise ValueError('Got an invalid data loader mode, ONLY SUPPORT: `train`, `test` and `valid`')

    def register_model(self, model, **kwargs):
        reuse = kwargs.get('reuse', False)
        is_trainable = kwargs.get('is_trainable', True)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)

        self.model = model.cuda()
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        if is_trainable:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)
            self.scaler = torch.cuda.amp.GradScaler()

        if reuse:
            self.load_ckpt(mode='batch', is_trainable=is_trainable)

    def save_ckpt(self, mode):
        if dist.get_rank() == 0:
            trainer_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "scaler": self.scaler.state_dict()
            }
            classifier_dict = {"state_dict": self.model.module.classifier.state_dict()}

            if mode == 'batch':
                torch.save(trainer_dict, osp.join(self.batch_ckpt_home, 'trainer.bin'))
                torch.save(classifier_dict, osp.join(self.batch_ckpt_home, 'classifier.bin'))
                self.model.module.embedding.lora_embedding.save_pretrained(self.batch_ckpt_home)
            elif mode in ['epoch', 'best']:
                torch.save(trainer_dict, osp.join(self.best_ckpt_home, 'trainer.bin'))
                torch.save(classifier_dict, osp.join(self.best_ckpt_home, 'classifier.bin'))
                self.model.module.embedding.lora_embedding.save_pretrained(self.best_ckpt_home)
            else:
                raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

    def load_ckpt(self, mode, is_trainable=False):
        if mode == 'batch':
            path = self.batch_ckpt_home
        elif mode in ['epoch', 'best']:
            path = self.best_ckpt_home
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

        if is_trainable:
            trainer_ckpt = torch.load(osp.join(path, 'trainer.bin'), map_location=torch.device('cuda'))
            self.optimizer.load_state_dict(trainer_ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(trainer_ckpt['lr_scheduler'])
            self.scaler.load_state_dict(trainer_ckpt['scaler'])

        # loading LoRA model
        self.model.module.embedding.lora_embedding.unload()
        self.model.module.embedding.lora_embedding.load_adapter(path, is_trainable=is_trainable, adapter_name='aaNER')

        # loading token classifier
        classifier_ckpt = torch.load(osp.join(path, 'classifier.bin'), map_location=torch.device('cuda'))
        classifier_state_dict = self.model.module.classifier.state_dict()
        classifier_trained_dict = {k: v for k, v in classifier_ckpt['state_dict'].items() if k in classifier_state_dict}
        classifier_state_dict.update(classifier_trained_dict)
        self.model.module.classifier.load_state_dict(classifier_state_dict)

    @abstractmethod
    def train(self, **kwargs):
        """train model"""

    @abstractmethod
    def inference(self, **kwargs):
        """inference model"""
