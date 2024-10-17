import xgboost as xgb
import torch
from classifiers.classifier import Classifier
from torch.utils import data
from pathlib import Path


class XGBoostClassifier(Classifier):
    """
    XGBoost-based classifier that extends the abstract Classifier class.

    Args:
        input_dim (int): Dimension of input features.
        output_dim (int): Dimension of output (number of classes).
        config (dict): Configuration dictionary with model hyperparameters.
    """

    def __init__(self, input_dim, output_dim, config: dict):
        """
        Initializes the XGBoostClassifier with the given configuration.

        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output (number of classes).
            config (dict): Configuration dictionary for model parameters.
        """
        super().__init__(input_dim, output_dim)

        # Initialize the XGBoost model with given parameters from config
        self.model = xgb.XGBClassifier(
            objective=config.get("objective", "multi:softmax"),  # Default: multi-class classification
            num_class=output_dim,  # Output dimension (number of classes)
            eta=config.get("eta", 0.08),  # Learning rate
            n_estimators=config.get("n_estimators", 80),  # Number of trees
            max_depth=config.get("max_depth", 10),  # Tree depth
            eval_metric=config.get("eval_metric", "mlogloss"),  # Evaluation metric
            verbosity=config.get("verbosity", 1)  # Verbosity level (training logs)
        )

        self.device = torch.device("cpu")

    def predict(self, test_data):
        """
        Predicts the labels for the provided test data.

        Args:
            test_data: Test data (feature matrix).

        Returns:
            Predicted labels as a NumPy array.
        """
        return torch.tensor(self.model.predict_proba(test_data.cpu()))

    def train(self, config, logger, train_ds, valid_ds, timestamp):
        """
        Trains the XGBoost classifier using the provided datasets.

        Args:
            config (dict): Training configuration dictionary.
            logger: Logger object for logging messages.
            train_ds: Training dataset.
            valid_ds: Validation dataset.
            timestamp (str): Timestamp or unique identifier for the training session.
        """
        # Ensure model saving folder exists
        Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

        # Prepare the training and validation data
        train_embeddings, train_labels, valid_embeddings, valid_labels = self.prepare_data(train_ds, valid_ds)

        # Train the XGBoost model
        self.model.fit(
            train_embeddings,
            train_labels,
            eval_set=[(valid_embeddings, valid_labels)],  # Evaluation on validation set
            verbose=True
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
            # Get embeddings and labels as PyTorch tensors
            embeddings = batch["emb"].to(self.device)  # Ensure embeddings are on the correct device (GPU/CPU)
            labels = torch.argmax(batch["labels"].squeeze(), dim=1).to(self.device)

            train_embeddings.append(embeddings)
            train_labels.append(torch.tensor(labels).to(self.device))

            # Concatenate all the embeddings and labels into single tensors
        train_embeddings = torch.cat(train_embeddings, dim=0).to(self.device)  # Concatenate along the first dimension
        train_labels = torch.cat(train_labels, dim=0).to(self.device)

        # Similarly, process validation data
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
        Loads a stored XGBoost model from the specified path.

        Args:
            classifier_path (str): Path to the saved model file.
        """
        self.model.load_model(classifier_path)

    def save_model(self, config, logger, timestamp):
        """
        Saves the trained XGBoost model to a file.

        Args:
            config (dict): Configuration dictionary.
            logger: Logger object to log the save event.
            timestamp (str): Timestamp to create unique file name.
        """
        model_filename = self.get_a_model_folder_path(config, f"xgboost_{timestamp}")
        self.model.save_model(model_filename)
        logger.warning(f"XGBoost model saved at: {model_filename}")
