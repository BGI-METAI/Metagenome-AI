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


class JSONConfigProvider:
    def get_config(self):
        raise NotImplementedError("Subclasses must implement get_config method")


class ESMConfigProvider(JSONConfigProvider):
    def get_config(self):
        with open("configs/config_esm.json") as file:
            return json.load(file)


class ProteinTransConfigProvider(JSONConfigProvider):
    def get_config(self):
        with open("config/config_protein_trans.json") as file:
            return json.load(file)


class ConfigProviderFactory:
    @staticmethod
    def get_config_provider(config_type):
        if config_type == "ESM":
            return ESMConfigProvider()
        elif config_type == "PTRANS":
            return ProteinTransConfigProvider()
        else:
            raise NotImplementedError("This type of config is not supported")


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
