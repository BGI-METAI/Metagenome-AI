from pathlib import Path
import json


class ConfigProviderFactory:
    @staticmethod
    def get_config_provider(config_file):
        with open(config_file) as file:
            return json.load(file)


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)