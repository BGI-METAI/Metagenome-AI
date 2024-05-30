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

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from safetensors import safe_open

from proteinNER.base_module import CustomNERDatasetCL
from proteinNER.base_module import CustomNERDataset
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
            timeout=timedelta(seconds=54000)
        )  # 1.5 hours
        self.accelerator = Accelerator(
            mixed_precision='fp16',# 混合精度，可能导致loss NAN
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
            # dataset = CustomNERDatasetCL(
            #     processed_sequence_label_pairs_path=data_files,
            #     tokenizer_model_name_or_path=model_name_or_path,
            #     legacy=legacy,
            #     do_lower_case=do_lower_case
            # )
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
            self.train_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=True,
            )
        elif mode == 'test':
            self.test_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
            )
        elif mode == 'valid':
            self.valid_loader = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
            )
        else:
            raise ValueError('Got an invalid data loader mode, ONLY SUPPORT: `train`, `test` and `valid`')

    def register_model(self, model, **kwargs):
        reuse = kwargs.get('reuse', False)
        is_trainable = kwargs.get('is_trainable', True)
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        mode = kwargs.get('mode', 'best')

        self.model = model
        if is_trainable:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=0.9)

        if reuse:
            self.load_ckpt(mode=mode, is_trainable=is_trainable)

    def save_ckpt(self, mode):
        if self.accelerator.main_process_first():
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            trainer_dict = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            classifier_dict = {"state_dict": unwrapped_model.classifier.state_dict()}

            if mode == 'batch':
                self.accelerator.save(trainer_dict, osp.join(self.batch_ckpt_home, 'trainer.bin'))
                self.accelerator.save(classifier_dict, osp.join(self.batch_ckpt_home, 'classifier.bin'))
                unwrapped_model.embedding.lora_embedding.save_pretrained(self.batch_ckpt_home)
            elif mode in ['epoch', 'best']:
                self.accelerator.save(trainer_dict, osp.join(self.best_ckpt_home, 'trainer.bin'))
                self.accelerator.save(classifier_dict, osp.join(self.best_ckpt_home, 'classifier.bin'))
                unwrapped_model.embedding.lora_embedding.save_pretrained(self.best_ckpt_home)
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

        # loading LoRA model
        lora_weight_tensor = {}
        with safe_open(osp.join(path, 'adapter_model.safetensors'), framework='pt') as file:
            for key in file.keys():
                lora_weight_tensor[key.replace('weight', 'default.weight')] = file.get_tensor(key)

        for name, weight in self.model.embedding.lora_embedding.named_parameters():
            if name not in lora_weight_tensor.keys():
                continue
            if weight.requires_grad:
                assert weight.data.size() == lora_weight_tensor[name].size(), f'Got an invalid key: `{name}`!'
                weight.data.copy_(lora_weight_tensor[name])
                if not is_trainable:
                    weight.requires_grad = False

        # loading token classifier
        classifier_ckpt = torch.load(osp.join(path, 'classifier.bin'), map_location=torch.device('cuda'))
        classifier_state_dict = self.model.classifier.state_dict()
        classifier_trained_dict = {k: v for k, v in classifier_ckpt['state_dict'].items() if k in classifier_state_dict}
        classifier_state_dict.update(classifier_trained_dict)
        self.model.classifier.load_state_dict(classifier_state_dict)

    @abstractmethod
    def train(self, **kwargs):
        """train model"""

    @abstractmethod
    def inference(self, **kwargs):
        """inference model"""
