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
import socket

import pandas as pd
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR, LinearLR

# from torcheval.metrics import MultilabelAccuracy
import wandb
import time

# Enable Multi-GPU training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from dataset import TSVDataset, MaxTokensLoader
from config import get_weights_file_path, ConfigProviderFactory
from utils.early_stopper import EarlyStopper
from utils.metrics import calc_metrics
import embeddings


def init_logger(config, timestamp):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        filename=f"{config['model_type']}_{config['program_mode']}_{timestamp}.log",
    )
    return logging.getLogger(__name__)


def init_wandb(model_folder, timestamp, model=None):
    # initialize wandb tracker
    wandb_output_dir = os.path.join(model_folder, "wandb_home")
    Path(wandb_output_dir).mkdir(parents=True, exist_ok=True)
    wandb.require("core")
    run = wandb.init(
        project="protein function annotation",
        notes=socket.gethostname(),
        name=f"prot_func_anno_{timestamp}",
        group="linear_classifier",
        dir=wandb_output_dir,
        job_type="training",
        reinit=True,
    )
    if model:
        wandb.watch(model, log="all")

    return run


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


class MLPClassifier(nn.Module):
    """A classification head used to output protein family (or other targets) probabilities

    Args:
        nn (_type_): _description_
    """

    def __init__(self, d_model, num_classes, hidden_sizes=None):
        super().__init__()
        if hidden_sizes is None:
            # Single fully connected layer for classification head
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Sequential(nn.Linear(d_model, num_classes))
        # Multiple hidden layers followed by a linear layer for classification head
        else:
            layers = []
            prev_size = d_model
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(prev_size, hidden_size))
                layers.append(nn.ReLU())
                prev_size = hidden_size
            layers.append(nn.Linear(prev_size, num_classes))
            self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = self.dropout(x)
        return self.classifier(x)


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


def store_embeddings(config, logger):
    world_size = torch.cuda.device_count()
    logger.info("Starts saving embeddings.")
    if "train" in config and config["train"] is not None:
        logger.info("Storing embedding from a train dataset.")
        mp.spawn(
            _store_embeddings,
            args=(config, logger, world_size, config["train"]),
            nprocs=world_size,
        )
    if "valid" in config and config["valid"] is not None:
        logger.info("Storing embedding from a validation dataset.")
        mp.spawn(
            _store_embeddings,
            args=(config, logger, world_size, config["valid"]),
            nprocs=world_size,
        )
    if "test" in config and config["test"] is not None:
        logger.info("Storing embedding from a test dataset.")
        mp.spawn(
            _store_embeddings,
            args=(config, logger, world_size, config["test"]),
            nprocs=world_size,
        )


def _store_embeddings(rank, config, logger, world_size, data_path):
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
                    llm.store_embeddings(batch, config["emb_dir"])
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(
                        "[REPROCESS] Cuda run out of mem, logging IDs that need to be re-processed"
                    )
                    logger.error(batch["protein_id"])
                    torch.cuda.empty_cache()
                    print("Cuda was out of memory, recovering...")

        end_time = time.time()
        print(f"Elapsed time: {(end_time - start_time)/60} min")
        # Resource cleanup
    finally:
        destroy_process_group()


def train_loop(config, logger, train_ds, valid_ds, timestamp):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader = data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=3,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = data.DataLoader(
        valid_ds,
        batch_size=config["batch_size"],
        drop_last=True,
    )

    # Or replace this part to be a part of the config as well
    llm = choose_llm(config)
    d_model = llm.get_embedding_dim()

    classifier = MLPClassifier(d_model, train_ds.get_number_of_labels()).to(device)

    run = init_wandb(config["model_folder"], timestamp, classifier)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    early_stopper = EarlyStopper(patience=4)
    # A 2-class problem can be modeled as:
    # - 2-neuron output with only one correct class: softmax + categorical_crossentropy
    # - 1-neuron output, one class is 0, the other is 1: sigmoid + binary_crossentropy
    loss_function = nn.CrossEntropyLoss()
    all_metrics_df = pd.DataFrame()

    for epoch in range(config["num_epochs"]):
        classifier.train()
        train_loss = 0
        for batch in train_dataloader:
            targets = batch["labels"].squeeze().to(device)
            embeddings = batch["emb"].to(device)

            optimizer.zero_grad()

            outputs = classifier(embeddings)

            loss = loss_function(outputs, targets)

            loss.backward()

            optimizer.step()

            train_loss += loss
        train_loss /= len(train_dataloader)
        scheduler.step()
        wandb.log({"train loss": train_loss})

        # Validation loop
        # first collect all outputs and then forward to metrics
        all_outputs = []
        all_targets = []
        classifier.eval()
        with torch.no_grad():
            multilabel_acc = val_loss = 0
            for batch in valid_dataloader:
                targets = batch["labels"].squeeze().to(device)
                embeddings = batch["emb"].to(device)
                outputs = classifier(embeddings)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                all_targets.append(targets.cpu())
                all_outputs.append(outputs.cpu())
                if valid_ds.get_number_of_labels() > 2:
                    # metric = MultilabelAccuracy(criteria="hamming")
                    metric = MultilabelAccuracy()
                    metric.update(outputs, torch.where(targets > 0, 1, 0))
                    multilabel_acc += metric.compute()

            all_targets = torch.cat(all_targets)
            all_outputs = torch.cat(all_outputs)
            epoch_metrics = calc_metrics(all_targets, all_outputs)
            all_metrics_df = pd.concat([all_metrics_df, epoch_metrics], ignore_index=True)

            val_loss = val_loss / len(valid_dataloader)
            wandb.log({"validation loss": val_loss})

            if valid_ds.get_number_of_labels() > 2:
                multilabel_acc = multilabel_acc / len(valid_dataloader)
                wandb.log({"validation multilabel accuracy": multilabel_acc})
            else:
                wandb.log({"validation accuracy": epoch_metrics['Accuracy'][0]})
                wandb.log({"validation f1_score": epoch_metrics['F1-score'][0]})
        logger.warning(
            f"Validation loss: {val_loss:.2f} Training loss: {train_loss:.2f}"
        )
        if early_stopper.early_stop(val_loss):
            logger.warning(f"Early stopping in epoch {epoch}...")
            break
        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            model_filename,
        )
        logging.warning(f"Finished epoch: {epoch + 1}")
        # log some metrics on batches and some metrics only on epochs
        # wandb.log({"batch": batch_idx, "loss": 0.3})
        # wandb.log({"epoch": epoch, "val_acc": 0.94})
    all_metrics_df.index = all_metrics_df.index + 1
    all_metrics_df = all_metrics_df.reset_index().rename(columns={'index': 'Epoch'})
    logger.info(f"\nValidation set scores per epoch \n {all_metrics_df.to_string(index=False)}")
    wandb.unwatch()
    run.finish()
    return classifier


def train_classifier_from_stored_single_gpu(config, logger):

    logger.info("Starts classifier training.")

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    train_ds = TSVDataset(config["train"], config["emb_dir"], "mean")
    valid_ds = TSVDataset(config["valid"], config["emb_dir"], "mean")
    test_ds = TSVDataset(config["test"], config["emb_dir"], "mean")

    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
    )

    if "classifier_path" in config and config["classifier_path"] is not None:
        llm = choose_llm(config)
        d_model = llm.get_embedding_dim()
        classifier = MLPClassifier(d_model, train_ds.get_number_of_labels()).to(device)
        print(f"Loading model: {config['classifier_path']}")
        state = torch.load(config["classifier_path"])
        classifier.load_state_dict(state["model_state_dict"])
    else:
        classifier = train_loop(config, logger, train_ds, valid_ds, timestamp)

    # Test loop
    all_outputs = []
    all_targets = []
    classifier.eval()
    multilabel_acc = 0
    with open(f"predictions_{timestamp}.tsv", "a") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(["protein_id", "prediction", "probability"])

    with torch.no_grad():
        for batch in test_dataloader:
            targets = batch["labels"].squeeze().to(device)
            embeddings = batch["emb"].to(device)
            outputs = classifier(embeddings)
            all_targets.append(targets.cpu())
            all_outputs.append(outputs.cpu())
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
                with open(f"predictions_{timestamp}.tsv", "a") as file:
                    writer = csv.writer(file, delimiter="\t")
                    content = list(
                        zip(
                            batch["protein_id"],
                            prediction_proba.indices.cpu().numpy(),
                            values_rounded,
                        )
                    )
                    writer.writerows(content)

        all_targets = torch.cat(all_targets)
        all_outputs = torch.cat(all_outputs)
        test_set_metrics = calc_metrics(all_targets, all_outputs)
        logger.info(f"\nTest set scores \n{test_set_metrics.to_string(index=False)}") 

        if test_ds.get_number_of_labels() > 2:
            multilabel_acc = multilabel_acc / len(test_dataloader)
            logger.info(f"[TEST SET] Multilabel accuracy: {multilabel_acc*100:.2f}%")
        else:
            logger.info(f"[TEST SET] Accuracy: {test_set_metrics['Accuracy'][0]*100:.2f}% F1: {test_set_metrics['F1-score'][0]*100:.2f}%")


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
    logger.info(json.dumps(config, indent = 4))

    #TODO :add finetuning before everything

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
        store_embeddings(config, logger)
    elif config["program_mode"] == valid_modes[1]:  # TRAIN_PREDICT_FROM_STORED
        train_classifier_from_stored_single_gpu(config, logger)
    elif config["program_mode"] == valid_modes[2]:  # RUN_ALL
        # mp.spawn(store_embeddings, args=(config, world_size), nprocs=world_size)
        store_embeddings(config, logger)
        train_classifier_from_stored_single_gpu(config, logger)
