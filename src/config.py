#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   config.py
@Time    :   2024/03/06 21:01:44
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

from pathlib import Path
import json


class ConfigProviderFactory:
    @staticmethod
    def get_config_provider(config_file):
        with open(config_file) as file:
            config = json.load(file)
            # set default values
            if "max_tokens" not in config:
                config["max_tokens"] = 2000
            return config


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_type = config["model_type"]
    model_filename = f"{model_type}_{model_basename}_{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
