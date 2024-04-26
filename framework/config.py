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
        "train": "~/MAI/Metagenome-AI/data/pfam_tiny.train.csv",
        "valid": "~/MAI/Metagenome-AI/data/pfam_tiny.valid.csv",
        "test": "~/MAI/Metagenome-AI/data/pfam_tiny.test.csv",
        "batch_size": 128,
        "num_epochs": 20,
        "lr": 1e-3,
        "emb_type": "PTRANS",
        "label": "label",
        "max_seq_len": 700 * 2,
        "sequence": "seq",
        "target": "family",
        "model_folder": "weights",
        "model_basename": "prot_model_",
        "preload": None,
        "prot_trans_model_name": "prot_t5_xl_uniref50",  
        # prot_bert, ProstT5, ProstT5_fp16,prot_t5_xl_uniref50, prot_t5_xl_half_uniref50-enc,
        # prot_t5_base_mt_uniref50, prot_t5_base_mt_uniref50, prot_bert_bfd_ss3, prot_bert_bfd_membrane,
        # prot_bert_bfd_localization, prot_t5_xxl_uniref50
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
