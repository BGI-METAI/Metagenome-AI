import csv
import datetime
import warnings
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
# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.distributed import init_process_group, destroy_process_group

from embedding import EsmEmbedding
from dataset import CustomDataset
from config import get_config, get_weights_file_path

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "caching_allocator"


# Initialize the PyTorch distributed backend
# This is essential even for single machine, multiple GPU training
# dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0)
# The distributed process group contains all the processes that can communicate and synchronize with each other.
# def ddp_setup(rank, world_size):
#     """
#     Args:
#         rank: Unique identifier of each process
#         world_size: Total number of processes
#     """
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "12356"
#     init_process_group(backend="nccl", rank=rank, world_size=world_size)
#     torch.cuda.set_device(rank)


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


def get_all_sentences(ds, lang):
    return ds[lang]


def get_ds(config):
    train_ds_raw = pd.read_csv(config["train"])
    le = LabelEncoder()
    le.fit(train_ds_raw[config["label"]])

    train_ds_raw["le_" + config["label"]] = le.transform(train_ds_raw[config["label"]])
    train_ds_raw = datasets.Dataset(pa.Table.from_pandas(train_ds_raw))

    if config["valid"] is not None and config["test"] is not None:
        val_ds_raw = pd.read_csv(config["valid"])
        test_ds_raw = pd.read_csv(config["test"])
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
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        # sampler=DistributedSampler(train_ds),
    )
    val_dataloader = data.DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False
    )
    test_dataloader = data.DataLoader(
        test_ds, batch_size=config["batch_size"], shuffle=False
    )

    return train_dataloader, val_dataloader, test_dataloader, le


def choose_llm(config):
    if config["emb_type"] == "ESM":
        return EsmEmbedding()
    else:
        raise NotImplementedError("This type of embedding is not supported")


def train_model(config):
    # def train_model(rank, config, world_size):
    # Define the device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rank = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device {device}")
    # ddp_setup(rank, world_size)

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, le = get_ds(config)

    model = choose_llm(config)
    model.to(rank)
    # model = DDP(model, device_ids=[rank])

    # model = DistributedDataParallel(model)
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    writer = SummaryWriter(config["experiment_name"] + timestamp)

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
    d_model = model.get_embedding_dim()
    num_classes = len(le.classes_)
    # classifier = Classifier(d_model, num_classes, [int((d_model + num_classes) / 2)])
    # Embedding layers transform the original data (AA sequence) into some semantic-aware
    # vector spaces. This is where all the architecture designs come in (e.g. attention, cnn, lstm etc.),
    # which are all far more superior than a simple FC for their chosen tasks. So if you have the capacity
    # of adding multiple FCs, why not just add another attention block? On the other hand, the embeddings
    # from a decent model should have large inter-class distance and small intra-class variance, which could
    #  easily be projected to their corresponding classes in a linear fashion, and a FC is more than enough.
    classifier = Classifier(d_model, num_classes).to(rank)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=config["lr"], eps=1e-9)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.7)
    # scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.5, total_iters=5)
    early_stopper = EarlyStopper(patience=5)

    # Training loop
    for epoch in range(config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch: {epoch:02d}")
        train_loss = 0
        classifier.train()
        for batch in batch_iterator:
            optimizer.zero_grad()

            embedding = model.get_embedding(batch, pooling="cls")
            classifier_output = classifier(embedding)

            target = batch[config["target"]].to(rank)

            loss = loss_fn(
                classifier_output,
                target,
            )
            train_loss += loss.item()
            batch_iterator.set_postfix({f"Training loss:": f"{loss.item():6.3f}"})

            loss.backward()
            optimizer.step()

        # before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        # after_lr = optimizer.param_groups[0]["lr"]
        # print("Epoch %d: Adam lr %.4f -> %.4f" % (epoch, before_lr, after_lr))

        train_loss = train_loss / len(train_dataloader)
        train_loss_list.append(train_loss)

        # Tensorboard
        writer.add_scalar("Training loss", train_loss, global_step)
        writer.flush()

        # Validation loop
        val_loss = 0
        classifier.eval()
        with torch.no_grad():
            acc = 0
            f1 = 0
            for batch in val_dataloader:
                embedding = model.get_embedding(batch)

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

        print(
            f"Accuracy: {acc:.2f} F1: {f1:.2f} Validation loss: {val_loss:.2f} Training loss: {train_loss:.2f}"
        )

        # Tensorboard
        writer.add_scalar("Validation loss", val_loss, global_step)
        writer.flush()

        if early_stopper.early_stop(val_loss):
            print("Early stopping...")
            break

        global_step += 1

        # Save model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )

    # Plot loss
    classifier.eval()
    predicted_labels = []
    with torch.no_grad():
        acc = 0
        f1 = 0
        for batch in test_dataloader:
            embedding = model.get_embedding(batch)

            classifier_output = classifier(embedding)
            target = batch[config["target"]].to(rank)
            pred = le.inverse_transform(torch.argmax(classifier_output, dim=1).cpu())

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
    df_result.to_csv(f"out_{config['emb_type']}.csv")

    acc = acc / len(test_dataloader)
    f1 = f1 / len(test_dataloader)

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    world_size = torch.cuda.device_count()
    # mp.spawn(train_model, args=(config, world_size), nprocs=world_size)
    train_model(config)
