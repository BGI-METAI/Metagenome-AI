#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   train.py
@Time    :   2024/03/06 19:55:37
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""
import argparse
import datetime
import logging
import json
import warnings
import os
from pathlib import Path
import csv

import pandas as pd
import torch
from torch.utils import data

from torcheval.metrics import MultilabelAccuracy
import wandb
import time

# Enable Multi-GPU training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from dataset import TSVDataset, MaxTokensLoader
from config import ConfigProviderFactory, choose_classifier, choose_llm
from utils.metrics import calc_metrics
from utils.umap_visual import plot_embeddings_umap


def init_logger(config, timestamp):
    """
    Creates a logger object that store logs in the log_dir
    Args:
        config (dict): Configuration file containing log_dir
        timestamp (string): Timestamp of the current run
    """
    # Check if 'log_dir' exists in config; if not, default to the current directory
    log_dir = config.get('log_dir', './logs/')
    # Ensure the directory exists, create it if not
    os.makedirs(log_dir, exist_ok=True)
    # Build the log file path
    log_file = os.path.join(log_dir, f"{config['model_type']}_{config['program_mode']}_{timestamp}.log")
    # Set up the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=log_file
    )

    return logging.getLogger(__name__)


def init_predictions_path(config, timestamp):
    """
    Creates a folder to store predictions and returns the path to the file
    Args:
        config (dict): Configuration file containing pred_dir
        timestamp (string): Timestamp of the current run
    """
    # Check if 'log_dir' exists in config; if not, default to the current directory
    pred_dir = config.get('pred_dir', './predictions/')
    # Ensure the directory exists, create it if not
    os.makedirs(pred_dir, exist_ok=True)
    # Build the tsv file path
    pred_file_path = os.path.join(pred_dir, f"predictions_{config['model_type']}_{config['classifier_type']}_{timestamp}.tsv")

    return pred_file_path


# Initialize the PyTorch distributed backend
# This is essential even for single machine, multiple GPU training
# dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0)
# The distributed process group contains all the processes that can communicate and synchronize with each other.
def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    # IP address that runs rank=0 process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def store_embeddings(config, logger, target_layer_index):
    world_size = torch.cuda.device_count()
    logger.info("Starts saving embeddings.")
    if "train" in config and config["train"] is not None:
        logger.info("Storing embedding from a train dataset.")
        mp.spawn(
            _store_embeddings,
            args=(config, logger, world_size, config["train"], target_layer_index),
            nprocs=world_size,
        )
    if "valid" in config and config["valid"] is not None:
        logger.info("Storing embedding from a validation dataset.")
        mp.spawn(
            _store_embeddings,
            args=(config, logger, world_size, config["valid"], target_layer_index),
            nprocs=world_size,
        )
    if "test" in config and config["test"] is not None:
        logger.info("Storing embedding from a test dataset.")
        mp.spawn(
            _store_embeddings,
            args=(config, logger, world_size, config["test"], target_layer_index),
            nprocs=world_size,
        )


def _store_embeddings(rank, config, logger, world_size, data_path, target_layer_index):
    try:
        ddp_setup(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        Path(config["emb_dir"]).mkdir(parents=True, exist_ok=True)

        ds = TSVDataset(data_path)
        max_tokens = config.get("max_tokens", 2000)
        chunk_size = len(ds) // world_size
        remainder = len(ds) % world_size

        start = 0
        indices = []
        for i in range(world_size):
            end = start + chunk_size + (1 if i < remainder else 0)
            indices.append((start, end))
            start = end

        # Each GPU gets its own part of the dataset
        dataloader = MaxTokensLoader(
            ds, start_ind=indices[rank][0], end_ind=indices[rank][1], max_tokens=max_tokens
        )

        llm = choose_llm(config)
        llm.to(device)

        dist.barrier()

        start_time = time.time()
        for batch in dataloader:
            try:
                with torch.no_grad():
                    llm.store_embeddings(batch, config["emb_dir"], target_layer_index)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(
                        "[REPROCESS] Cuda run out of mem, logging IDs that need to be re-processed"
                    )
                    logger.error(batch["protein_id"])
                    torch.cuda.empty_cache()
                    print("Cuda was out of memory, recovering...")

        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time) / 60} min")
        # Resource cleanup
    finally:
        destroy_process_group()


def train_classifier_from_stored_single_gpu(config, logger, target_layer_index):
    logger.info("Starts classifier training.")

    llm = choose_llm(config)
    
    if target_layer_index is not None and not (1 <= target_layer_index <= len(llm.model.layers)):
        logger.warning(f"Invalid target_layer_index: {target_layer_index}. It will be treated as None.")
        target_layer_index = None

    layer_str = f"mean_hidden_layer{target_layer_index}" if target_layer_index else "mean"
    
    train_ds = TSVDataset(config["train"], config["emb_dir"], layer_str)
    valid_ds = TSVDataset(config["valid"], config["emb_dir"], layer_str)
    test_ds = TSVDataset(config["test"], config["emb_dir"], layer_str)

    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
    )

    d_model = llm.get_embedding_dim()
    num_labels = train_ds.get_number_of_labels()
    classifier = choose_classifier(config, input_dimensions=d_model, output_dimensions=num_labels)

    if "classifier_path" in config and config["classifier_path"] is not None:
        print(f"Loading model: {config['classifier_path']}")
        classifier.load_stored(config['classifier_path'])
    else:
        classifier.train(config, logger, train_ds, valid_ds, timestamp)

    # Test loop
    all_outputs = []  # List to collect model outputs
    all_targets = []  # List to collect targets/labels
    multilabel_acc = 0

    predictions_path = init_predictions_path(config, timestamp)

    with open(predictions_path, "a") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["protein_id", "prediction", "probability"])

    for batch in test_dataloader:
        targets = batch["labels"].squeeze().cpu()  # Move labels to CPU
        embeddings = batch["emb"].cpu()  # Move embeddings to CPU
        outputs = classifier.predict(embeddings)  # Predict using classifier

        # Append the current batch targets and outputs to the lists
        all_targets.append(targets)  # Append to Python list
        all_outputs.append(outputs)  # Append to Python list

        if test_ds.get_number_of_labels() > 2:
            metric = MultilabelAccuracy()
            metric.update(outputs, torch.where(targets > 0, 1, 0))
            multilabel_acc += metric.compute()
            # pass
            # code to save results in .tsv
        else:
            proba = torch.nn.functional.softmax(outputs)
            prediction_proba = torch.max(proba, dim=1)
            values_rounded = [
                round(p, 4) for p in prediction_proba.values.cpu().numpy()
            ]
            with open(predictions_path, "a") as file:
                writer = csv.writer(file, delimiter="\t")
                content = list(
                    zip(
                        batch["protein_id"],
                        prediction_proba.indices.cpu().numpy(),
                        values_rounded,
                    )
                )
                writer.writerows(content)

    # Concatenate all the targets and outputs at the end, after the loop
    all_targets = torch.cat(all_targets)
    all_outputs = torch.cat(all_outputs)

    if test_ds.get_number_of_labels() > 2:
        multilabel_acc = multilabel_acc / len(test_dataloader)
        logger.info(f"[TEST SET] Multilabel accuracy: {multilabel_acc * 100:.2f}%")
    else:
        # Calculate metrics on the test set
        test_set_metrics = calc_metrics(all_targets, all_outputs)
        logger.info(f"\nTest set scores \n{test_set_metrics.to_string(index=False)}")
        logger.info(
            f"[TEST SET] Accuracy: {test_set_metrics['Accuracy'][0] * 100:.2f}% F1: {test_set_metrics['F1-score'][0] * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module that contains training and evaluation. Will be separated."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="Type of embedding to be used",
        type=str,
        required=False,
        default="ESM",
    )

    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    config = ConfigProviderFactory.get_config_provider(args.config_path)
    wandb.login(key=config["wandb_key"])
    world_size = torch.cuda.device_count()

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    logger = init_logger(config, timestamp)
    logger.info(json.dumps(config, indent=4))

    target_layer_index = config.get("target_layer_index", None)

    # TODO :add finetuning before everything

    valid_modes = ["ONLY_STORE_EMBEDDINGS", "TRAIN_PREDICT_FROM_STORED", "RUN_ALL"]
    if config["program_mode"] not in valid_modes:
        print(
            f"Invalid program mode: {config['program_mode']}. Turned on default mode of eperation [RUN_ALL]."
        )
        config["program_mode"] = "RUN_ALL"
    else:
        print(f"Program mode is: {config['program_mode']}.")

    if config["program_mode"] == valid_modes[0]:  # ONLY_STORE_EMBEDDINGS
        # mp.spawn(store_embeddings, args=(config, world_size), nprocs=world_size)
        store_embeddings(config, logger, target_layer_index)
    elif config["program_mode"] == valid_modes[1]:  # TRAIN_PREDICT_FROM_STORED
        train_classifier_from_stored_single_gpu(config, target_layer_index)
    elif config["program_mode"] == valid_modes[2]:  # RUN_ALL
        # mp.spawn(store_embeddings, args=(config, world_size), nprocs=world_size)
        store_embeddings(config, logger, target_layer_index)
        train_classifier_from_stored_single_gpu(config, logger, target_layer_index)
    
    #plot_embeddings_umap(config)