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
import os


class ConfigProviderFactory:
    @staticmethod
    def get_config_provider(config_file):
        with open(config_file) as file:
            config = json.load(file)
            return config


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


class ConfigsGenerator:
    DIRECTORIES = ["embeddings", "classifiers", "predictions", "logs", "configs"]
    RUN_MODES = ["create_embeddings", "train_classifiers"]

    def __init__(self, config):
        """
        Initializes ConfigsGenerator with configuration settings and prepares directory structure.

        Args:
            config (dict): Dictionary containing the main configuration, including 'run_name'.
        """
        self.run_name = config.get('run_name', 'default')
        self.run_directory = f"./runs/{self.run_name}"

        # Extract configuration lists for models, datasets, and classifiers
        self.models = config.get("models", [])
        self.datasets = config.get("datasets", [])
        self.class_heads = config.get("classifiers", [])
        # Config Files
        self.wandb_key = config.get("wandb_key", None)
        self.configs = []

    def get_configs(self):
        return self.configs

    def get_directory_path(self, directory):
        """
        Gets the full path for a specific directory under the main run directory.

        Args:
            directory (str): The subdirectory name within the run directory.

        Returns:
            str: Full path to the specified subdirectory.
        """
        return os.path.join(self.run_directory, directory)

    def get_model_base_name(self, model, database, classifier=None):
        """
        Creates a base name for the model configuration file based on model, database, and optionally classifier.

        Args:
            model (dict): Dictionary with model parameters.
            database (dict): Dictionary with database parameters.
            classifier (dict, optional): Dictionary with classifier parameters.

        Returns:
            str: A formatted string to serve as the base name for the configuration file.
        """
        if classifier is None:
            return f"{model['model_name']}_{database['dataset_name']}"
        else:
            return f"{model['model_name']}_{database['dataset_name']}_{classifier['classifier_name']}"

    def create_model_path(self, directory, model, database, classifier=None):
        """
        Creates and returns the directory path for storing model-related files.

        Args:
            directory (str): The main directory under which model paths are created.
            model (dict): Dictionary with model parameters.
            database (dict): Dictionary with database parameters.
            classifier (dict, optional): Dictionary with classifier parameters.

        Returns:
            str: Path where model-related files are stored.
        """
        model_name = self.get_model_base_name(model, database, classifier)
        model_path = os.path.join(self.get_directory_path(directory), model_name)
        os.makedirs(model_path, exist_ok=True)
        return model_path

    def create_starting_directories(self):
        """Creates the main directory and required subdirectories for the run."""
        os.makedirs(self.run_directory, exist_ok=True)
        for folder in self.DIRECTORIES:
            os.makedirs(os.path.join(self.run_directory, folder), exist_ok=True)

    def write_config_file(self, config, model, dataset, classifier=None):
        """
        Writes a configuration dictionary to a JSON file based on the mode (embedding or classifier).

        Args:
            config (dict): Configuration dictionary to write to file.
            model (dict): Model parameters, used for naming the file.
            dataset (dict): Dataset params.
            classifier (dict, optional): Classifier parameters, if creating a classifier config.
        """
        # Determine mode and filename based on whether a classifier is provided
        config_mode = "create_embeddings" if classifier is None else "train_classifiers"
        model_base_name = self.get_model_base_name(model, dataset, classifier)
        config_file_name = f"{model_base_name}.json"

        # Create the path for the config file and ensure the directory exists
        config_directory = os.path.join(self.get_directory_path("configs"), config_mode)
        os.makedirs(config_directory, exist_ok=True)
        config_file_path = os.path.join(config_directory, config_file_name)

        # Write the config dictionary to the JSON file
        with open(config_file_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)

        self.configs.append({
            "run_mode": config_mode,
            "config_path": config_file_path,
            "model_base_name": model_base_name,
            "env": model.get("env")
        })

    def create_config_file(self, model, dataset, classifier=None):
        """
        Creates a configuration file for either embeddings or classifier training based on input.

        Args:
            model (dict): Dictionary with model parameters.
            dataset (dict): Dictionary with dataset parameters.
            classifier (dict, optional): Dictionary with classifier parameters if training a classifier.
        """
        config = {}
        # Unpack model and dataset into config
        config.update(model)
        config.update(dataset)
        if classifier:
            config.update(classifier)

        # Set run mode and directories for embeddings or classifier training
        config["program_mode"] = "TRAIN_PREDICT_FROM_STORED" if classifier else "ONLY_STORE_EMBEDDINGS"
        config["emb_dir"] = self.create_model_path("embeddings", model, dataset)
        config["log_dir"] = self.get_directory_path("logs")
        config["wandb_key"] = self.wandb_key
        if classifier:
            config["pred_dir"] = self.get_directory_path("predictions")
            config["model_folder"] = self.create_model_path("classifiers", model, dataset, classifier)
            config["model_basename"] = self.get_model_base_name(model, dataset, classifier)

        # Write the finalized configuration file
        self.write_config_file(config, model, dataset, classifier)

    def generate(self):
        """
        Generates the config files for each model, dataset, and classifier combination.

        """
        self.create_starting_directories()  # Initialize required directories for the run.

        # Create directories for embeddings and classifiers configurations
        os.makedirs(os.path.join(self.run_directory, os.path.join("configs", "create_embeddings")), exist_ok=True)
        os.makedirs(os.path.join(self.run_directory, os.path.join("configs", "train_classifiers")), exist_ok=True)

        # Iterate through the parameters and create configuration files
        for model in self.models:
            for dataset in self.datasets:
                self.create_config_file(model, dataset)
                for classifier in self.class_heads:
                    self.create_config_file(model, dataset, classifier)
