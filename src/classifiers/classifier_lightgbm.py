import lightgbm as lgb
import torch
from classifiers.classifier import Classifier
from torch.utils import data
from pathlib import Path

class LightGBMClassifier(Classifier):
    """
    LightGBM-based classifier that extends the abstract Classifier class.

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output (number of classes).
        config (dict): Configuration dictionary with model hyperparameters.
    """

    def __init__(self, input_dim, output_dim, config: dict):
        """
        Initializes the LightGBMClassifier with the given configuration.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output (number of classes).
            config (dict): Configuration dictionary for model parameters.
        """
        super().__init__(input_dim, output_dim)

        # Initialize the LightGBM model with parameters from config
        self.model = lgb.LGBMClassifier(
            objective=config.get("objective", "multiclass"),  # Default objective for multiclass classification
            num_class=output_dim,  # Number of classes
            learning_rate=config.get("learning_rate", 0.05),  # Learning rate
            n_estimators=config.get("n_estimators", 150),  # Number of boosting rounds
            max_depth=config.get("max_depth", -1),  # Tree depth
            num_leaves=config.get("num_leaves", 31),  # Number of leaves per tree
            feature_fraction=config.get("feature_fraction", 0.9),  # Subsampling ratio of features
            bagging_fraction=config.get("bagging_fraction", 0.8),  # Subsampling ratio of samples
            bagging_freq=config.get("bagging_freq", 5),  # Frequency for bagging
            early_stopping_round=config.get("early_stop", 10),  # Early stopping rounds
            verbosity=1
        )

        self.device = torch.device("cpu")

    def predict(self, test_data):
        """
        Predicts the labels for the provided test data.

        Args:
            test_data: Test data (feature matrix).

        Returns:
            Predicted labels as a PyTorch tensor.
        """
        return torch.tensor(self.model.predict_proba(test_data.cpu()))

    def train(self, config, logger, train_ds, valid_ds, timestamp):
        """
        Trains the LightGBM classifier using the provided datasets.

        Args:
            config (dict): Training configuration dictionary.
            logger: Logger object for logging messages.
            train_ds (TSVDataset): Training dataset.
            valid_ds (TSVDataset): Validation dataset.
            timestamp (str): Timestamp or unique identifier for the training session.
        """
        # Ensure model saving folder exists
        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

        # Prepare the training and validation data
        train_embeddings, train_labels, valid_embeddings, valid_labels = self.prepare_data(train_ds, valid_ds)

        # Train the LightGBM model
        self.model.fit(
            train_embeddings,
            train_labels,
            eval_set=[(valid_embeddings, valid_labels)]
        )

        # Save the trained model
        self.save_model(config, logger, timestamp)

    def prepare_data(self, train_ds, valid_ds):
        """
        Prepares the data by extracting embeddings and labels from the datasets.

        Args:
            train_ds: Training dataset.
            valid_ds: Validation dataset.

        Returns:
            Tuple: (train_embeddings, train_labels, valid_embeddings, valid_labels)
        """
        # Use DataLoader to extract batches
        train_dataloader = data.DataLoader(
            train_ds,
            batch_size=64,
            shuffle=True,
            num_workers=3,
            pin_memory=True,
            drop_last=True,
        )
        valid_dataloader = data.DataLoader(
            valid_ds,
            batch_size=64,
            drop_last=True,
        )

        # Containers for embeddings and labels
        train_embeddings, train_labels = [], []
        valid_embeddings, valid_labels = [], []

        # Collect training data
        for batch in train_dataloader:
            embeddings = batch["emb"].to(self.device)
            labels = torch.argmax(batch["labels"].squeeze(), dim=1).to(self.device)

            train_embeddings.append(embeddings)
            train_labels.append(torch.tensor(labels).to(self.device))

        train_embeddings = torch.cat(train_embeddings, dim=0).to(self.device)
        train_labels = torch.cat(train_labels, dim=0).to(self.device)

        # Process validation data
        for batch in valid_dataloader:
            embeddings = batch["emb"].to(self.device)
            labels = torch.argmax(batch["labels"].squeeze(), dim=1).to(self.device)
            valid_embeddings.append(embeddings)
            valid_labels.append(torch.tensor(labels).to(self.device))

        valid_embeddings = torch.cat(valid_embeddings, dim=0).to(self.device)
        valid_labels = torch.cat(valid_labels, dim=0).to(self.device)

        return train_embeddings, train_labels, valid_embeddings, valid_labels

    def load_stored(self, classifier_path):
        """
        Loads a stored LightGBM model from the specified path.

        Args:
            classifier_path (str): Path to the saved model file.
        """
        self.model = lgb.Booster(model_file=classifier_path)

    def save_model(self, config, logger, timestamp):
        """
        Saves the trained LightGBM model to a file.

        Args:
            config (dict): Configuration dictionary.
            logger: Logger object to log the save event.
            timestamp (str): Timestamp to create unique file name.
        """
        model_filename = self.get_a_model_folder_path(config, f"lightgbm_{timestamp}")
        self.model.booster_.save_model(model_filename)
        logger.warning(f"LightGBM model saved at: {model_filename}")
