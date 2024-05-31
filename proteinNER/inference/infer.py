#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/5/30 9:19
# @Author  : zhangchao
# @File    : infer.py
# @Email   : zhangchao5@genomics.cn
import socket
import os
import os.path as osp
import torch
import pickle

from abc import abstractmethod
from dataclasses import field
from datetime import timedelta, datetime
from pathlib import Path
from typing import List
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm import tqdm

from proteinNER.base_module.dataset import InferDataset
from proteinNER.inference.framework import ProtTrans4AAClassifier


class BaseInfer:
    model: ProtTrans4AAClassifier = field(default=None, metadata={'help': 'protein sequence amino acid classifier'})
    data_loader: DataLoader = field(default=None, metadata={'help': 'data loader'})
    batch_size: int = field(default=1, metadata={'help': 'batch size'})
    accelerator: Accelerator = field(default=None)
    base_model_name_or_path: str = field(default=None)
    amino_acid_classifier_model_path: str = field(default=None)
    label2ids: List = []
    id2labels: List = field(default=None)

    def __init__(self, **kwargs):
        self.base_model_name_or_path = kwargs.get('model_name_or_path')
        self.batch_size = kwargs.get('batch_size')
        self.amino_acid_classifier_model_path = kwargs.get('amino_acid_classifier_model_path')
        self.output_home = kwargs.get('output_home')

        set_seed(kwargs.get('seed', 42))
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=10800)
        )
        self.accelerator = Accelerator(
            mixed_precision='fp16',
            log_with='wandb',
            kwargs_handlers=[process_group_kwargs]
        )
        self.wandb_home = self.register_dir(
            parent_path=self.output_home,
            folder='tracker'
        )
        self.pkls_home = self.register_dir(
            parent_path=self.output_home,
            folder='predict'
        )

    def register_dir(self, parent_path, folder):
        new_path = osp.join(parent_path, folder)
        with self.accelerator.main_process_first():
            Path(new_path).mkdir(parents=True, exist_ok=True)
        return new_path

    def register_tracker(
            self,
            user_name,
            project_name,
            group,
            timestamp=datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    ):
        self.accelerator.init_trackers(
            project_name=project_name,
            init_kwargs={
                'wandb': {
                    'entity': user_name,
                    'notes': socket.gethostname(),
                    'name': f'ProtTransAminoAcidClassifier_{timestamp}',
                    'group': group,
                    'dir': self.wandb_home,
                    'job_type': 'inference',
                    'reinit': True
                }
            }
        )

    def _validate(self):
        pass

    def register_data(
            self,
            data_files,
    ):
        dataset = InferDataset(
            processed_sequence_label_pairs_path=data_files,
            tokenizer_model_name_or_path=self.base_model_name_or_path,
            legacy=False,
            do_lower_case=False
        )

        self.data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False
        )

    def register_model(self, num_classes, n_header):
        self.model = ProtTrans4AAClassifier(
            base_model_name_or_path=self.base_model_name_or_path,
            num_classes=num_classes,
            n_header=n_header
        )
        self.load_ckpt()

    def load_ckpt(self):
        header_ckpt_home = [osp.join(self.amino_acid_classifier_model_path, name) for name in
                            sorted(os.listdir(self.amino_acid_classifier_model_path)) if
                            os.path.isdir(osp.join(self.amino_acid_classifier_model_path, name))]

        state_dict = self.model.state_dict()

        assert len(header_ckpt_home) == len(self.model.multiheader)
        for idx, ckpt_home in enumerate(header_ckpt_home):
            # load id2label dictionary
            self.label2ids.append(pickle.load(open(osp.join(ckpt_home, 'label2id.pkl'), 'rb')))

            trans_dict = torch.load(osp.join(ckpt_home, 'transition.bin'), map_location=torch.device('cuda'))
            trans_trained_dict = {k.replace(k, f'multiheader.{idx}.transition.{k}'): v for k, v in
                                  trans_dict['state_dict'].items()
                                  if k.replace(k, f'multiheader.{idx}.transition.{k}') in state_dict}
            state_dict.update(trans_trained_dict)

            crf_dict = torch.load(osp.join(ckpt_home, 'crf.bin'), map_location=torch.device('cuda'))
            crf_trained_dict = {k.replace(k, f'multiheader.{idx}.crf.{k}'): v for k, v in crf_dict['state_dict'].items()
                                if
                                k.replace(k, f'multiheader.{idx}.crf.{k}') in state_dict}
            state_dict.update(crf_trained_dict)
        self.model.load_state_dict(state_dict)
        self.id2labels = [{str(v): k if k == 'O' else k.split('-')[1] for k, v in sub.items()} for sub in
                          self.label2ids]

    @abstractmethod
    def inference(self, **kwargs):
        """inference amino acid"""


class ProtTransAminoAcidClassifierInfer(BaseInfer):
    def __init__(self, **kwargs):
        super(ProtTransAminoAcidClassifierInfer, self).__init__(**kwargs)
        self.register_data(data_files=kwargs.get('pairs_files'))
        self.register_model(num_classes=kwargs.get('num_classes'),
                            n_header=kwargs.get('num_header'))

    @torch.no_grad()
    def inference(self):
        model = self.accelerator.prepare_model(self.model)
        data_loader = self.accelerator.prepare_data_loader(self.data_loader)
        model.eval()
        batch_iter = tqdm(data_loader, desc=f'PID: {self.accelerator.process_index}')

        for idx, sample in enumerate(batch_iter):
            input_ids, attention_mask, protein_ids = sample
            predict = model(input_ids, attention_mask)
            self.post_process(predict, protein_ids)

    def post_process(self, predict, protein_ids, *, threshold=0.65):
        rearrange = [list(row) for row in zip(*predict)]
        for idx, row in enumerate(rearrange):
            temp = {}
            for jdx, subhead in enumerate(row):
                tag, prob = max(subhead['probability'].items(), key=lambda x: x[1])
                temp[f'header{jdx}'] = (self.id2labels[jdx][tag], prob)
            with open(osp.join(self.pkls_home, f'{protein_ids[idx]}.pkl'), 'wb') as fp:
                pickle.dump(temp, fp)