from abc import ABC, abstractmethod
from pathlib import Path


class Classifier(ABC):
    """
    Abstract base class for classifiers.

    Args:
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output (number of classes/labels).
    """

    def __init__(self, input_dim, output_dim):
        """
        Initialize the classifier with input and output dimensions,
        and set the device (GPU if available, otherwise CPU).

        Args:
            input_dim (int): Dimension of the input features.
            output_dim (int): Dimension of the output (number of classes).
        """
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def train(self, config, logger, train_ds, valid_ds, timestamp):
        """
        Abstract method for training logic, must be implemented by subclasses.

        Args:
            config (dict): Training configuration parameters.
            logger: Logger object for logging during training.
            train_ds (Dataset): Training dataset.
            valid_ds (Dataset): Validation dataset.
            timestamp (str): Timestamp or unique identifier for the training session.
        """
        pass

    @abstractmethod
    def predict(self, test_data):
        """
        Abstract method for prediction logic, must be implemented by subclasses.

        Args:
            test_data (Dataset): Test dataset or data to run inference on.
        """
        pass

    @abstractmethod
    def load_stored(self, classifier_path):
        """
        Abstract method to load a stored/pretrained classifier model.

        Args:
            classifier_path (str): Path to the stored classifier model.
        """
        pass

    def get_a_model_folder_path(self, config, epoch: str):
        model_folder = config["model_folder"]
        model_basename = config["model_basename"]
        model_type = config["model_type"]
        model_filename = f"{model_type}_{model_basename}_{epoch}.pt"
        return str(Path(".") / model_folder / model_filename)
