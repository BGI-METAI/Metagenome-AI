import argparse
import datetime
import logging
import warnings
import os
from pathlib import Path
import csv
import socket

from matplotlib import pyplot as plt
import datasets
import pandas as pd
import pyarrow as pa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.utils import data
from torch.optim.lr_scheduler import StepLR, LinearLR
# from torcheval.metrics import MultilabelAccuracy
from tqdm import tqdm
import wandb
import time

# Enable Multi-GPU training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from embed.embedding_dataset import TSVDataset, MaxTokensLoader, CSVDataset, MaxTokensLoaderLocal
from utils.config import get_weights_file_path, ConfigProviderFactory
from utils.memory_check import check_gpu_used_memory

from model.model_classify import Classifier
from read_embedding import CustomDataset

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

def train_classifier_from_stored_single_gpu(config):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    train_ds = CustomDataset(config["train"], config["emb_dir"], "mean")
    valid_ds = CustomDataset(config["valid"], config["emb_dir"], "mean")
    test_ds = CustomDataset(config["test"], config["emb_dir"], "mean")

    # train_ds = CSVDataset("/home/share/huadjyin/home/wangshengfu/06_esm/data_AMP/data_train.csv")
    # valid_ds = CSVDataset("/home/share/huadjyin/home/wangshengfu/06_esm/data_AMP/data_val.csv")
    # test_ds = CSVDataset("/home/share/huadjyin/home/wangshengfu/06_esm/data_AMP/data_test.csv")

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
    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
    )

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    # logger = init_logger(timestamp)

    # 应只有2 种标签
    classifier = Classifier(d_model, train_ds.get_number_of_labels()).to(device)
    # run = init_wandb(config["model_folder"], timestamp, classifier)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    early_stopper = EarlyStopper(patience=4)
    # A 2-class problem can be modeled as:
    # - 2-neuron output with only one correct class: softmax + categorical_crossentropy
    # - 1-neuron output, one class is 0, the other is 1: sigmoid + binary_crossentropy
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(config["num_epochs"]):
        classifier.train()
        train_loss = 0
        for batch in train_dataloader:
            targets = batch["label"].squeeze().to(device)
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
        classifier.eval()
        with torch.no_grad():
            acc = f1 = multilabel_acc = val_loss = 0
            for batch in valid_dataloader:
                targets = batch["labels"].squeeze().to(device)
                embeddings = batch["emb"].to(device)
                outputs = classifier(embeddings)
                loss = loss_function(outputs, targets)
                val_loss += loss.item()
                if valid_ds.get_number_of_labels() > 2:
                    # metric = MultilabelAccuracy(criteria="hamming")
                    metric = MultilabelAccuracy()
                    metric.update(outputs, torch.where(targets > 0, 1, 0))
                    multilabel_acc += metric.compute()
                else:
                    acc += accuracy_score(
                        torch.argmax(targets, dim=1).cpu(),
                        torch.argmax(outputs, dim=1).cpu(),
                    )
                    f1 += f1_score(
                        torch.argmax(targets, dim=1).cpu(),
                        torch.argmax(outputs, dim=1).cpu(),
                    )

            val_loss = val_loss / len(valid_dataloader)
            wandb.log({"validation loss": val_loss})

            if valid_ds.get_number_of_labels() > 2:
                multilabel_acc = multilabel_acc / len(valid_dataloader)
                wandb.log({"validation multilabel accuracy": multilabel_acc})
            else:
                acc = acc / len(valid_dataloader)
                f1 = f1 / len(valid_dataloader)
                wandb.log({"validation accuracy": acc})
                wandb.log({"validation f1_score": f1})
        logger.warning(
            f"[VALIDATION SET] Accuracy: {acc:.2f} F1: {f1:.2f} Validation loss: {val_loss:.2f} Training loss: {train_loss:.2f}"
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
        logging.warning(f"Finished epoch: {epoch}")
        # log some metrics on batches and some metrics only on epochs
        # wandb.log({"batch": batch_idx, "loss": 0.3})
        # wandb.log({"epoch": epoch, "val_acc": 0.94})

    run.finish()