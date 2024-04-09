#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   config.py
@Time    :   2024/03/06 21:01:44
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
'''

from pathlib import Path


def get_config():
    return {
        "train":
        "Metagenome-AI/data/pfam_tiny.train.csv",
        "valid":
        "Metagenome-AI/data/pfam_tiny.valid.csv",
        "test": "Metagenome-AI/data/pfam_tiny.test.csv",
        "batch_size": 64,
        "num_epochs": 20,
        "lr": 1e-3,
        "emb_type": "PVEC",
        "label": "label",
        "max_seq_len": 700 * 2,
        "sequence": "seq",
        "target": "family",
        "model_folder": "weights",
        "model_basename": "pvec_model_",
        "preload": None,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
