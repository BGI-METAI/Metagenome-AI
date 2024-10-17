from torch import nn
from torch.optim.lr_scheduler import StepLR

from classifiers.classifier import Classifier
from config import get_weights_file_path
from pathlib import Path
from torch.utils import data
from utils.wandb import init_wandb
import torch
import pandas as pd
from torcheval.metrics import MultilabelAccuracy
from utils.early_stopper import EarlyStopper
from utils.metrics import calc_metrics
import logging
import wandb


class MLPModel(nn.Module):
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


class MLPClassifier(Classifier):
    """A Multi-Layer Perceptron (MLP) classifier.

    Inherits from the base `Classifier` class. This classifier uses an MLP model
    for predicting the probabilities of protein family memberships based on embeddings.

    Args:
        input_dim (int): Dimensionality of the input embeddings.
        output_dim (int): Number of classes (output labels).
        config (dict): Configuration dictionary containing training settings and hyperparameters.
    """
    def __init__(self, input_dim, output_dim, config:dict):
        super().__init__(input_dim, output_dim)
        self.hidden_dims = config.get("hidden_layers", None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MLPModel(input_dim, output_dim, self.hidden_dims).to(self.device)

    def train(self, config, logger, train_ds, valid_ds, timestamp):
        """Trains the MLPClassifier on the provided training dataset and validates using the validation dataset.

        Args:
            config (dict): Configuration dictionary containing training settings.
            logger (logging.Logger): Logger object for logging training details.
            train_ds (Dataset): Training dataset.
            valid_ds (Dataset): Validation dataset.
            timestamp (str): Timestamp for organizing model saving.
        """
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

        run = init_wandb(config["model_folder"], timestamp, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], eps=1e-9)
        scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
        early_stopper = EarlyStopper(patience=4)
        # A 2-class problem can be modeled as:
        # - 2-neuron output with only one correct class: softmax + categorical_crossentropy
        # - 1-neuron output, one class is 0, the other is 1: sigmoid + binary_crossentropy
        loss_function = nn.CrossEntropyLoss()
        all_metrics_df = pd.DataFrame()

        for epoch in range(config["num_epochs"]):
            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                targets = batch["labels"].squeeze().to(self.device)
                embeddings = batch["emb"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(embeddings)

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
            self.model.eval()
            with torch.no_grad():
                multilabel_acc = val_loss = 0
                for batch in valid_dataloader:
                    targets = batch["labels"].squeeze().to(self.device)
                    embeddings = batch["emb"].to(self.device)
                    outputs = self.model(embeddings)
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
                    "model_state_dict": self.model.state_dict(),
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

    def load_stored(self, classifier_path):
        """Loads the model weights from a saved checkpoint.

        Args:
            classifier_path (str): Path to the saved model checkpoint file.
        """
        state = torch.load(classifier_path)
        self.model.load_state_dict(state["model_state_dict"])

    def predict(self, test_data):
        self.model.eval()
        test_data = test_data.to(self.device)  # Move to the appropriate device
        with torch.no_grad():
            predictions = self.model(test_data)
        return predictions.cpu()  # Ensure predictions are returned on CPU
