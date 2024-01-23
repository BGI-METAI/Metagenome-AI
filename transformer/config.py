from pathlib import Path


def get_config():
    return {
        "input": "transformer/pfam_tiny_1579.train.csv",
        "batch_size": 16,
        "num_epochs": 2,
        "lr": 1e-4,
        "seq_len": 520,
        "d_model": 64,
        "src": "original",
        "tgt": "corrupted",
        "model_folder": "weights",
        "model_basename": "pmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
