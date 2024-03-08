import datetime
import logging
import warnings
import os
from pathlib import Path

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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# # Enable Multi-GPU training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

from framework.config import get_weights_file_path, get_config
from framework.dataset import CustomDataset
from framework.esm2 import Esm2Embeddings


def init_logger():
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    logging.basicConfig(
        format="%(name)-12s %(levelname)-8s %(message)s",
        level=logging.INFO,
        filename=f"{timestamp}.log",
    )


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
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_ds(config):
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
        test_ds_raw["le_" + config["label"]] = le.transform(test_ds_raw[config["label"]])
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
    # val_dataloader = data.DataLoader(
    #    val_ds, batch_size=config["batch_size"], shuffle=False
    # )
    # test_dataloader = data.DataLoader(
    #    test_ds, batch_size=config["batch_size"], shuffle=False
    # )

    return train_dataloader, val_ds, test_ds, le


def choose_llm(config):
    if config["emb_type"] == "ESM":
        return Esm2Embeddings()
    else:
        raise NotImplementedError("This type of embedding is not supported")


def train_model_test(rank, config, world_size):
    ddp_setup(rank, world_size)
    own = torch.tensor(dist.get_rank()).cuda()

    # Gather tensors on process 0 (GPU 0) using `gather`
    gathered_tensor = [torch.zeros_like(own) for _ in range(world_size)]
    if rank == 0:
        dist.gather(own, gathered_tensor, dst = 0)
    else:
        dist.gather(own)

    flag = torch.tensor(0).cuda()
    print(flag.dtype)

    if rank == 0:
        # check if should be stopped
        flag = torch.tensor(3).cuda()

    # print("Before rank ", rank, " ", flag.item())
    dist.all_reduce(flag, op=dist.ReduceOp.SUM)
    if flag.item() == 1:
        print("Should be stopped")
    else:
        print("Still good")

    # print("After rank: ", rank, " ", flag.item())
    # All processes print the gathered tensor on process 0 (GPU 0)
    print(f'Process {dist.get_rank()}: {own.item()} -> {gathered_tensor}')


def train_classifier(rank, config, world_size):
    print(f"GPU RANK: {rank}")
    # Define the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if rank == 0:
        init_logger()

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ddp_setup(rank, world_size)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_ds, test_ds, le = get_ds(config)
    if rank == 0:
        val_dataloader = data.DataLoader(val_ds,
                batch_size=config["batch_size"], shuffle=False)
        test_dataloader = data.DataLoader(test_ds,
        batch_size=config["batch_size"], shuffle=False)
    else:
        val_dataloader = None
        test_dataloader = None

    llm = choose_llm(config)
    llm.to(device)

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    if config["tensorboard"]:
        writer = SummaryWriter(config["experiment_name"] + timestamp)

    optimizer = torch.optim.Adam(llm.parameters(), lr=config["lr"], eps=1e-9)
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

    train_loss_list = []
    val_loss_list = []
    acc_list = []
    f1_list = []

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
    #  easily be projected to their corresponding classes in a linear fashion, and a FC is more than enough.
    # classifier = Classifier(d_model, num_classes).to(rank)
    classifier = Classifier(d_model, num_classes).to(rank)
    classifier = DDP(classifier, device_ids=[rank])

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
    # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=5)
    early_stopper = EarlyStopper(patience=4)
    stop_flag = torch.tensor(0).to(rank)

    # Training loop
    for epoch in range(config["num_epochs"]):
        if stop_flag.item() == 1:
            # All GPUs got signal from rank 0 to break
            logging.warning(f"Early stopping in epoch {epoch}. Training finished...")
            break

        train_dataloader.sampler.set_epoch(epoch)
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch on rank {rank}: {epoch:02d}")

        mem_used = torch.cuda.max_memory_allocated() / (1024**3)
        logging.info(f"GPU_{rank} used max mem in epoch {epoch}: {mem_used:.2f} GB")
        torch.cuda.reset_peak_memory_stats()

        train_loss = torch.tensor(0.0).to(rank)
        classifier.train()
        for batch in batch_iterator:
            optimizer.zero_grad()

            embedding = llm.get_embedding(batch, pooling="mean")
            classifier_output = classifier(embedding)

            target = batch[config["target"]].to(rank)

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

        # This loss if from one GPU (rank)
        train_loss = train_loss / len(train_dataloader)
        # Gather loss from each process
        # There will be world_size number of losses that
        # will be averaged and plotted as a final loss of certain epoch
        train_loss_gather = [torch.zeros_like(train_loss) for _ in range(world_size)]
        if rank == 0:
            print(f"Gathered in rank {rank}")
            dist.gather(train_loss, gather_list=train_loss_gather)
        else:
            print(f"Gathered in rank {rank}")
            dist.gather(train_loss)

        # Calculate average loss
        train_loss_avg = torch.stack(train_loss_gather).mean().item()
        train_loss_list.append(train_loss_avg)

        # Tensorboard
        if config["tensorboard"] and rank == 0:
            writer.add_scalar("Training loss", train_loss, global_step)
            writer.flush()

        if rank == 0:
            # Validation loop
            print("Entering validation only on rank 0")
            val_loss = 0
            classifier.eval()
            with torch.no_grad():
                acc = 0
                f1 = 0
                for batch in val_dataloader:
                    embedding = llm.get_embedding(batch)

                    classifier_output = classifier(embedding)
                    target = batch[config["target"]].to(rank)

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
            acc_list.append(acc)
            f1 = f1 / len(val_dataloader)
            f1_list.append(f1)
            val_loss = val_loss / len(val_dataloader)
            val_loss_list.append(val_loss)

            logging.warning(
                f"Accuracy: {acc:.2f} F1: {f1:.2f} Validation loss: {val_loss:.2f} Training loss: {train_loss:.2f}"
            )

            # Tensorboard
            if config["tensorboard"]:
                writer.add_scalar("Validation loss", val_loss, global_step)
                writer.flush()

            if early_stopper.early_stop(val_loss):
                print(f"Early stopping, rank {rank} setting stop_flag...")
                stop_flag = torch.tensor(1).cuda()

        dist.all_reduce(stop_flag, op=dist.ReduceOp.SUM)
        print(f"Reduced stop_flag on rank {rank}")

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

    if rank == 0:
        # Test loop
        print(f"Entering test eval on rank {rank}")
        classifier.eval()
        predicted_labels = []
        acc = 0
        f1 = 0
        with torch.no_grad():
            for batch in test_dataloader:
                embedding = llm.get_embedding(batch)

                classifier_output = classifier(embedding)
                target = batch[config["target"]].to(rank)
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

        print(f"[TEST SET] Accuracy: {acc:.2f} F1: {f1:.2f}")

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(val_loss_list, label="Validation")
        ax.plot(train_loss_list, label="Training")
        ax.set_ylabel("loss")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper left")
        ax.set_frame_on(False)
        plt.tight_layout()
        plt.savefig(f"loss_{rank}.png")
        # Plot scores
        fig, ax = plt.subplots()
        ax.plot(acc_list, label="Accuracy")
        ax.plot(f1_list, label="F1_score")
        ax.set_ylabel("score")
        ax.set_xlabel("epoch")
        ax.legend(loc="upper left")
        ax.set_frame_on(False)
        plt.tight_layout()
        plt.savefig(f"score_{rank}.png")
        # Resource cleanup
    destroy_process_group()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    world_size = torch.cuda.device_count()
    # world_size = 2
    mp.spawn(train_classifier, args=(config, world_size), nprocs=world_size)
    # train_model(config)
