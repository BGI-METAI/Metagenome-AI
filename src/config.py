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
import embeddings
import classifiers


class ConfigProviderFactory:
    @staticmethod
    def get_config_provider(config_file):
        with open(config_file) as file:
            config = json.load(file)
            return config


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_type = config["model_type"]
    model_filename = f"{model_type}_{model_basename}_{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)


def choose_llm(config):
    """Select a pretrained model that produces embeddings

    Args:
        config (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Embedding: A subclass of class Embedding
    """
    if config["model_type"] == "ESM":
        return embeddings.EsmEmbedding(config)
    elif config["model_type"] == "ESM3":
        return embeddings.Esm3Embedding(config)
    elif config["model_type"] == "PTRANS":
        return embeddings.ProteinTransEmbedding(config)
    elif config["model_type"] == "PVEC":
        return embeddings.ProteinVecEmbedding()
    raise NotImplementedError("This type of embedding is not supported")


def choose_classifier(config, input_dimensions, output_dimensions):
    """Select a classifier model to train on embeddings

    Args:
        config (dict): config file containing additional parameters for the classifier and classifier_type
        input_dimensions (int): number of input dimensions for the classifier
        output_dimensions (int): number of output dimensions for the classifier, number of unique classes in dataset

    Raises:
        NotImplementedError: _description_

    Returns:
        Embedding: A subclass of class Embedding
    """
    if config["classifier_type"] == "MLP":
        return classifiers.MLPClassifier(input_dim=input_dimensions, output_dim=output_dimensions, config=config)
    elif config["classifier_type"] == "XGBoost":
        return classifiers.XGBoostClassifier(input_dim=input_dimensions, output_dim=output_dimensions, config=config)

    raise NotImplementedError("This type of classifier is not supported")
