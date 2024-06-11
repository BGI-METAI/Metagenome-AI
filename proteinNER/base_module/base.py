#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/28 10:06
# @Author  : zhangchao
# @File    : base.py
# @Email   : zhangchao5@genomics.cn
import os.path as osp
import socket
import torch

from abc import ABC, abstractmethod
from dataclasses import field
from pathlib import Path
from datetime import timedelta, datetime
from typing import Optional, Union
from functools import partial

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from proteinNER.base_module import CustomNERDataset


class BaseTrainer(ABC):
    """
    Base Trainer

    Args:
        output_home: str, the output home path
        save_in_batch: bool, whether to save checkpoint in batch
        seed: int, the random seed
        k: int, the gradient accumulation steps
        username: str, the wandb username
        project: str, the wandb project name
        group: str, the wandb group name
    """
    model: Optional[Union[torch.nn.Module]] = field(default=None, metadata={
        "help": "protein sequence AA classifier model"})
    train_loader: DataLoader = field(default=None, metadata={"help": "train loader"})
    test_loader: DataLoader = field(default=None, metadata={"help": "test loader"})
    valid_loader: DataLoader = field(default=None, metadata={"help": "valid loader"})
    optimizer: Optimizer = field(default=None, metadata={"help": "optimizer for training model"})
    lr_scheduler: LRScheduler = field(default=None, metadata={"help": "Learning rate decay course schedule"})
    batch_size: int = field(default=1, metadata={"help": "batch size"})
    loss_weight: float = field(default=1., metadata={"help": "loss weight"})
    max_epoch: int = field(default=100, metadata={"help": "max epoch"})
    learning_rate: float = field(default=1e-3, metadata={"help": "learning rate"})
    is_trainable: bool = field(default=True, metadata={"help": "whether the model to be train or not"})
    reuse: bool = field(default=False, metadata={"help": "whether the model parameters to be reuse or not"})
    accelerator: Accelerator = field(default=None)

    def __init__(self, **kwargs):
        set_seed(kwargs.get('seed', 42))
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(days=1, seconds=10800)
        )
        self.accelerator = Accelerator(
            mixed_precision='fp16',
            log_with='wandb',
            gradient_accumulation_steps=kwargs.get('k', 1),
            kwargs_handlers=[process_group_kwargs]
        )

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

    def register_dir(self, parent_path, folder):
        new_path = osp.join(parent_path, folder)

        with self.accelerator.main_process_first():
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

        self.accelerator.init_trackers(
            project_name=project_name,
            config={
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'loss_weight': self.loss_weight,
                'max_epoch': self.max_epoch
            },
            init_kwargs={
                'wandb': {
                    'entity': user_name,
                    'notes': socket.gethostname(),
                    'name': f'ProteinSeqNER_{timestamp}',
                    'group': group,
                    'dir': self.wandb_home,
                    'job_type': 'training',
                    'reinit': True
                }
            }
        )

    def register_dataset(
            self,
            data_files,
            label2id_path,
            mode,
            dataset_type='class',
            **kwargs
    ):
        """
        :param data_files: pickle files path list of protein sequence
        :param label2id_path:
        :param mode: data loader type, optional, only support `train`, `test` and `valid`
        :param dataset_type: dataset type, optional, only support `class` and `embed`

        :return:
        """
        self.batch_size = kwargs.get('batch_size', 1)
        model_name_or_path = kwargs.get('model_name_or_path')
        legacy = kwargs.get('legacy', False)
        do_lower_case = kwargs.get('do_lower_case', False)
        is_valid = kwargs.get('is_valid', False)

        if dataset_type == 'class':
            dataset = CustomNERDataset(
                processed_sequence_label_pairs_path=data_files,
                label2id_path=label2id_path,
                tokenizer_model_name_or_path=model_name_or_path,
                legacy=legacy,
                do_lower_case=do_lower_case
            )
        elif dataset_type == 'embed':
            raise NotImplementedError
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `class` and `embed`')

        partial_collate_fn = partial(dataset.collate_fn, is_valid=is_valid)

        if mode == 'train':
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=partial_collate_fn,
                shuffle=True,
            )
        elif mode == 'test':
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=partial_collate_fn,
                shuffle=False,
            )
        else:
            raise ValueError('Got an invalid data loader mode, ONLY SUPPORT: `train` and `test`!')

    def register_model(self, model, **kwargs):
        reuse = kwargs.get('reuse', False)
        is_trainable = kwargs.get('is_trainable', True)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        mode = kwargs.get('mode', 'best')
        lr_decay_step = kwargs.get('lr_decay_step', 2)
        lr_decay_gamma = kwargs.get('lr_decay_gamma', 0.99)

        self.model = model
        if is_trainable:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, lr_decay_step, gamma=lr_decay_gamma)

        if reuse:
            self.load_ckpt(mode=mode, is_trainable=is_trainable)

    def save_ckpt(self, mode):
        if self.accelerator.main_process_first():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            trainer_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            transition_state_dict = {"state_dict": unwrapped_model.transition.state_dict()}
            crf_state_dict = {"state_dict": unwrapped_model.crf.state_dict()}

            if mode == 'batch':
                self.accelerator.save(trainer_dict, osp.join(self.batch_ckpt_home, 'trainer.bin'))
                self.accelerator.save(transition_state_dict, osp.join(self.batch_ckpt_home, 'transition.bin'))
                self.accelerator.save(crf_state_dict, osp.join(self.batch_ckpt_home, 'crf.bin'))

            elif mode in ['epoch', 'best']:
                self.accelerator.save(trainer_dict, osp.join(self.best_ckpt_home, 'trainer.bin'))
                self.accelerator.save(transition_state_dict, osp.join(self.best_ckpt_home, 'transition.bin'))
                self.accelerator.save(crf_state_dict, osp.join(self.best_ckpt_home, 'crf.bin'))
            else:
                raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

    def load_ckpt(self, mode, is_trainable=False):
        if mode == 'batch':
            ckpt_home = self.batch_ckpt_home
        elif mode in ['epoch', 'best']:
            ckpt_home = self.best_ckpt_home
        else:
            raise ValueError('Got an invalid dataset mode, ONLY SUPPORT: `batch`, `epoch` or `best`')

        if is_trainable:
            trainer_ckpt = torch.load(osp.join(ckpt_home, 'trainer.bin'), map_location=torch.device('cuda'))
            self.optimizer.load_state_dict(trainer_ckpt['optimizer'])
            self.lr_scheduler.load_state_dict(trainer_ckpt['lr_scheduler'])

        state_dict = self.model.state_dict()

        transition_dict = torch.load(osp.join(ckpt_home, 'transition.bin'), map_location=torch.device('cuda'))
        transition_trained_dict = {k.replace(k, f'transition.{k}'): v for k, v in transition_dict['state_dict'].items()
                                   if k.replace(k, f'transition.{k}') in state_dict}
        state_dict.update(transition_trained_dict)

        crf_dict = torch.load(osp.join(ckpt_home, 'crf.bin'), map_location=torch.device('cuda'))
        crf_trained_dict = {k.replace(k, f'crf.{k}'): v for k, v in crf_dict['state_dict'].items() if
                            k.replace(k, f'crf.{k}') in state_dict}
        state_dict.update(crf_trained_dict)

        self.model.load_state_dict(state_dict)

    def print_trainable_parameters(self):
        total = 0
        trainable = 0

        for k, v in self.model.named_parameters():
            total += v.numel()
            if v.requires_grad:
                trainable += v.numel()
        self.accelerator.log({
            'Trainable Params': f'trainable params: {trainable} || all params: {total} || trainable%: {trainable / total:.15f}'
        })

    @abstractmethod
    def train(self, **kwargs):
        """train model"""

    @abstractmethod
    def inference(self, **kwargs):
        """inference model"""
