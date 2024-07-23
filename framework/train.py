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
from torcheval.metrics import MultilabelAccuracy
from tqdm import tqdm
import wandb
import time

# Enable Multi-GPU training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from dataset import CustomDataset, TSVDataset, MaxTokensLoader
from config import get_weights_file_path, ConfigProviderFactory
from utils import check_gpu_used_memory

try:
    from embedding_esm import EsmEmbedding
except ImportError:
    print("You are missing some of the libraries for ESM")
try:
    from embedding_protein_trans import ProteinTransEmbedding
except ImportError:
    print("You are missing some of the libraries for ProteinTrans")
try:
    from embedding_protein_vec import ProteinVecEmbedding
except ImportError:
    print("You are missing some of the libraries for ProteinVec")


def init_logger(timestamp):
    logging.basicConfig(
        format="%(name)-12s %(levelname)-8s %(message)s",
        level=logging.INFO,
        filename=f"{timestamp}.log",
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


class Classifier(nn.Module):
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
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_datasets(config):
    train_ds_raw = pd.read_csv(config["train"])
    le = LabelEncoder()
    le.fit(train_ds_raw[config["label"]])

    train_ds_raw["le_" + config["label"]] = le.transform(train_ds_raw[config["label"]])
    train_ds_raw = datasets.Dataset(pa.Table.from_pandas(train_ds_raw))

    if config["valid"] is not None and config["test"] is not None:
        val_ds_raw = pd.read_csv(config["valid"])
        val_ds_raw["le_" + config["label"]] = le.transform(val_ds_raw[config["label"]])
        val_ds_raw = datasets.Dataset(pa.Table.from_pandas(val_ds_raw))

        test_ds_raw = pd.read_csv(config["test"])
        test_ds_raw["le_" + config["label"]] = le.transform(
            test_ds_raw[config["label"]]
        )
        test_ds_raw = datasets.Dataset(pa.Table.from_pandas(test_ds_raw))
    else:
        # 80% training 10% validation 10% test split
        train_ds_raw, val_ds_raw, test_ds_raw = data.random_split(
            train_ds_raw, [0.8, 0.1, 0.1]
        )

    train_ds = CustomDataset(train_ds_raw, config)
    val_ds = CustomDataset(val_ds_raw, config)
    test_ds = CustomDataset(test_ds_raw, config)

    train_dataloader = data.DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=3,
        pin_memory=True,
        persistent_workers=True,
        sampler=DistributedSampler(train_ds),
    )

    return train_dataloader, val_ds, test_ds, le


def choose_llm(config):
    """Select a pretrained model that produces embeddings

    Args:
        config (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Embedding: A subclass of class Embedding
    """
    if config["emb_type"] == "ESM":
        return EsmEmbedding()
    elif config["emb_type"] == "PTRANS":
        if (
            "prot_trans_model_path" not in config.keys()
            or config["prot_trans_model_path"] is None
        ):
            return ProteinTransEmbedding(model_name=config["prot_trans_model_name"])
        else:
            return ProteinTransEmbedding(
                model_name=config["prot_trans_model_path"], read_from_files=True
            )
    elif config["emb_type"] == "PVEC":
        return ProteinVecEmbedding()
    else:
        raise NotImplementedError("This type of embedding is not supported")


def train_model_test(rank, world_size):
    """TODO Will be removed, used for debugging

    Args:
        rank (int): _description_
        world_size (int): _description_
    """
    # ddp_setup(rank, world_size)
    # own = torch.tensor(dist.get_rank()).cuda()

    # # Gather tensors on process 0 (GPU 0) using `gather`
    # gathered_tensor = [torch.zeros_like(own) for _ in range(world_size)]
    # if rank == 0:
    #     dist.gather(own, gathered_tensor, dst = 0)
    # else:
    #     dist.gather(own)

    # flag = torch.tensor(0).cuda()
    # print(flag.dtype)

    # if rank == 0:
    #     # check if should be stopped
    #     flag = torch.tensor(3).cuda()

    # print("Before rank ", rank, " ", flag.item())
    # dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    # if flag.item() == 1:
    #     print("Should be stopped")
    # else:
    #     print("Still good")
    # print("After rank: ", rank, " ", flag.item())

    # # All processes print the gathered tensor on process 0 (GPU 0)
    # print(f'Process {dist.get_rank()}: {own.item()} -> {gathered_tensor}')
    pass


def store_embeddings(config):
    world_size = torch.cuda.device_count()
    if "train" in config and config["train"] is not None:
        mp.spawn(
            _store_embeddings,
            args=(config, world_size, config["train"]),
            nprocs=world_size,
        )
    if "valid" in config and config["valid"] is not None:
        mp.spawn(
            _store_embeddings,
            args=(config, world_size, config["valid"]),
            nprocs=world_size,
        )
    if "test" in config and config["test"] is not None:
        mp.spawn(
            _store_embeddings,
            args=(config, world_size, config["test"]),
            nprocs=world_size,
        )


def _store_embeddings(rank, config, world_size, data_path):
    try:
        ddp_setup(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        Path(config["emb_dir"]).mkdir(parents=True, exist_ok=True)

        ds = TSVDataset(data_path)
        max_tokens = config["max_tokens"]
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
            ds, max_tokens, start_ind=indices[rank][0], end_ind=indices[rank][1]
        )

        llm = choose_llm(config)
        llm.to(device)

        timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        logger = init_logger(timestamp)

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

    classifier = Classifier(d_model, train_ds.get_number_of_labels()).to(device)

    run = init_wandb(config["model_folder"], timestamp, classifier)
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
    wandb.unwatch()
    run.finish()
    return classifier


def train_classifier_from_stored_single_gpu(config):
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    logger = init_logger(timestamp)

    train_ds = TSVDataset(config["train"], config["emb_dir"], "mean")
    valid_ds = TSVDataset(config["valid"], config["emb_dir"], "mean")
    test_ds = TSVDataset(config["test"], config["emb_dir"], "mean")

    test_dataloader = data.DataLoader(
        test_ds,
        batch_size=config["batch_size"],
    )

    if "model_path" in config and config["model_path"] is not None:
        llm = choose_llm(config)
        d_model = llm.get_embedding_dim()
        classifier = Classifier(d_model, train_ds.get_number_of_labels()).to(device)
        print(f"Loading model: {config['model_path']}")
        state = torch.load(config["model_path"])
        classifier.load_state_dict(state["model_state_dict"])
    else:
        classifier = train_loop(config, logger, train_ds, valid_ds, timestamp)

    # Test loop
    classifier.eval()
    acc = f1 = multilabel_acc = 0
    with open(f"predictions_{timestamp}.csv", "a") as file:
        writer = csv.writer(file, delimiter=" ")
        writer.writerow(["protein_id", "prediction", "probability"])

    with torch.no_grad():
        for batch in test_dataloader:
            embeddings = batch["emb"].to(device)
            outputs = classifier(embeddings)
            if test_ds.get_number_of_labels() > 2:
                pass
                # code to save results in .csv
            else:
                proba = torch.nn.functional.softmax(outputs)
                prediction_proba = torch.max(proba, dim=1)
                values_rounded = [
                    round(p, 4) for p in prediction_proba.values.cpu().numpy()
                ]
                with open(f"predictions_{timestamp}.csv", "a") as file:
                    writer = csv.writer(file, delimiter="\t")
                    content = list(
                        zip(
                            batch["protein_id"],
                            prediction_proba.indices.cpu().numpy(),
                            values_rounded,
                        )
                    )
                    writer.writerows(content)

        if test_ds.get_number_of_labels() > 2:
            multilabel_acc = multilabel_acc / len(test_dataloader)
            logger.info(f"[TEST SET] Multilabel accuracy: {acc*100:.2f}%")
        else:
            acc = acc / len(test_dataloader)
            f1 = f1 / len(test_dataloader)
            logger.info(f"[TEST SET] Accuracy: {acc*100:.2f}% F1: {f1*100:.2f}%")


def train_classifier(rank, config, world_size):
    """Classifier training based on LLM embeddings

    Args:
        rank (int): ID of GPU
        config (dict): (Hyper)parameters for training
        world_size (int): Total number of available GPUs
    """
    try:
        ddp_setup(rank, world_size)
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

        train_dataloader, val_ds, test_ds, le = get_datasets(config)

        if rank == 0:
            val_dataloader = data.DataLoader(
                val_ds, batch_size=config["batch_size"], shuffle=False
            )
            test_dataloader = data.DataLoader(
                test_ds, batch_size=config["batch_size"], shuffle=False
            )
        else:
            val_dataloader = None
            test_dataloader = None

        llm = choose_llm(config)
        llm.to(device)

        initial_epoch = 0
        global_step = 0

        if config["preload"]:
            model_filename = get_weights_file_path(config, config["preload"])
            print(f"Preloading model: {model_filename}")
            state = torch.load(model_filename)
            initial_epoch = state["epoch"] + 1
            global_step = state["global_step"]
            optimizer.load_state_dict(state["optimizer_state_dict"])

        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1).to(rank)

        # Fine-tune the classifier using the embeddings to predict protein families
        # d_model = config["d_model"]
        # Each model should return its embedding dimension
        d_model = llm.get_embedding_dim()
        num_classes = len(le.classes_)
        # classifier = Classifier(d_model, num_classes, [int((d_model + num_classes) / 2)])
        # Embedding layers transform the original data (AA sequence) into some semantic-aware
        # vector spaces. This is where all the architecture designs come in (e.g. attention, cnn, lstm etc.),
        # which are all far more superior than a simple FC for their chosen tasks. So if you have the capacity
        # of adding multiple FCs, why not just add another attention block? On the other hand, the embeddings
        # from a decent model should have large inter-class distance and small intra-class variance, which could
        # easily be projected to their corresponding classes in a linear fashion, and a FC is more than enough.
        # classifier = Classifier(d_model, num_classes).to(rank)
        classifier = Classifier(d_model, num_classes).to(rank)
        classifier = DDP(classifier, device_ids=[rank])

        if rank == 0:
            timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
            logger = init_logger(timestamp)
            init_wandb(config["model_folder"], timestamp, classifier)

        # To wait for wandb to get initialized
        dist.barrier()

        optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], eps=1e-9)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
        # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=5)
        early_stopper = EarlyStopper(patience=4)
        # Flag to signal all processes/GPUs to stop training because of overfitting
        stop_flag = torch.tensor(0).to(rank)

        # Training loop
        for epoch in range(config["num_epochs"]):
            if stop_flag.item() == 1:
                # All GPUs got signal from rank 0 to end training
                if rank == 0:
                    logger.warning(
                        f"Early stopping in epoch {epoch}. Training finished..."
                    )
                break

            train_dataloader.sampler.set_epoch(epoch)
            batch_iterator = tqdm(
                train_dataloader, desc=f"Processing epoch on rank {rank}: {epoch:02d}"
            )

            # mem_used = torch.cuda.max_memory_allocated() / (1024**3)
            # logging.info(f"GPU_{rank} used max mem in epoch {epoch}: {mem_used:.2f} GB")
            # torch.cuda.reset_peak_memory_stats()

            train_loss = torch.tensor(0.0).to(rank)
            classifier.train()
            for batch in batch_iterator:
                optimizer.zero_grad()

                embedding = llm.get_embedding(batch)
                classifier_output = classifier(embedding)

                target = batch["target"].to(rank)

                loss = loss_fn(
                    classifier_output,
                    target,
                )
                train_loss += loss
                batch_iterator.set_postfix({f"Training loss:": f"{loss.item():6.3f}"})

                loss.backward()
                optimizer.step()
                # print(f"Break rank {rank} finish epoch {epoch}")
                # break
                # alloc = torch.cuda.memory_allocated() / (1024**2)
                # reserved = torch.cuda.memory_reserved() / (1024**2)
                # logging.warning(
                #    f"GPU_{rank} Allocated: {alloc} [MiB] Reserved(alloc+cached): {reserved} [MiB]"
                # )

            # before_lr = optimizer.param_groups[0]["lr"]
            scheduler.step()
            # after_lr = optimizer.param_groups[0]["lr"]
            # print("Epoch %d: Adam lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

            # This loss is from one GPU (GPU_rank)
            train_loss = train_loss / len(train_dataloader)
            # Gather loss from each process
            # There will be world_size number of losses that
            # will be averaged and plotted as a final loss of certain epoch
            train_loss_gather = [
                torch.zeros_like(train_loss) for _ in range(world_size)
            ]
            if rank == 0:
                dist.gather(train_loss, gather_list=train_loss_gather)
            else:
                dist.gather(train_loss)

            # Calculate average loss
            train_loss_avg = torch.stack(train_loss_gather).mean().item()
            if rank == 0:
                wandb.log({"train_loss": train_loss_avg})

            # Validation loop
            if rank == 0:
                logger.info("Entering validation only on rank 0")
                val_loss = 0
                classifier.eval()
                with torch.no_grad():
                    acc = 0
                    f1 = 0
                    for batch in val_dataloader:
                        embedding = llm.get_embedding(batch)

                        classifier_output = classifier(embedding)
                        target = batch["target"].to(rank)

                        loss = loss_fn(
                            classifier_output,
                            target,
                        )
                        val_loss += loss.item()

                        # Accuracy and f1
                        acc += accuracy_score(
                            target.cpu(), torch.argmax(classifier_output, dim=1).cpu()
                        )
                        f1 += f1_score(
                            target.cpu(),
                            torch.argmax(classifier_output, dim=1).cpu(),
                            average="macro",
                        )
                acc = acc / len(val_dataloader)
                f1 = f1 / len(val_dataloader)
                val_loss = val_loss / len(val_dataloader)
                wandb.log({"val_loss": val_loss})
                wandb.log({"accuracy": acc})
                wandb.log({"f1_score": f1})

                logger.warning(
                    f"Accuracy: {acc:.2f} F1: {f1:.2f} Validation loss: {val_loss:.2f} Training loss: {train_loss:.2f}"
                )

                if early_stopper.early_stop(val_loss):
                    print(f"Early stopping, rank {rank} setting stop_flag...")
                    stop_flag = torch.tensor(1).cuda()

            dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)

            global_step += 1

            # Save model at the end of every epoch
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": classifier.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "global_step": global_step,
                },
                model_filename,
            )
            logging.warning(f"Finished epoch: {epoch}")

        # Test loop
        if rank == 0:
            logger.info("Entering evaluation on test set only on rank 0")
            classifier.eval()
            predicted_labels = []
            acc = 0
            f1 = 0
            with torch.no_grad():
                for batch in test_dataloader:
                    embedding = llm.get_embedding(batch)

                    classifier_output = classifier(embedding)
                    target = batch["target"].to(rank)
                    pred = le.inverse_transform(
                        torch.argmax(classifier_output, dim=1).cpu()
                    )

                    # Accuracy and f1
                    acc += accuracy_score(
                        target.cpu(), torch.argmax(classifier_output, dim=1).cpu()
                    )
                    f1 += f1_score(
                        target.cpu(),
                        torch.argmax(classifier_output, dim=1).cpu(),
                        average="macro",
                    )
                    predicted_labels.extend(pred)

            # Saving results to csv
            df_result = pd.DataFrame(predicted_labels)
            df_result.to_csv(f"out_{config['emb_type']}_{timestamp}.csv")

            acc = acc / len(test_dataloader)
            f1 = f1 / len(test_dataloader)

            logger.info(f"[TEST SET] Accuracy: {acc:.2f} F1: {f1:.2f}")

        # Resource cleanup
    finally:
        destroy_process_group()


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
        store_embeddings(config)
    elif config["program_mode"] == valid_modes[1]:  # TRAIN_PREDICT_FROM_STORED
        train_classifier_from_stored_single_gpu(config)
    elif config["program_mode"] == valid_modes[2]:  # RUN_ALL
        # mp.spawn(store_embeddings, args=(config, world_size), nprocs=world_size)
        store_embeddings(config)
        train_classifier_from_stored_single_gpu(config)
