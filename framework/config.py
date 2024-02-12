from pathlib import Path


def get_config():
    return {
        "input": "../transformer/pfam_tiny_1579.train.csv",
        "batch_size": 16,
        "num_epochs": 10,
        "d_model": 480,
        "lr": 1e-4,
        "emb_type": "ESM",
        "label": "original",
        "model_folder": "weights",
        "model_basename": "pmodel_",
        "preload": None,
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
