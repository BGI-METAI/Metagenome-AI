from pathlib import Path


def get_config():
    return {
        # "train": '/goofys/projects/MAI/data/brain-genomics-public/pfam.test.csv',
        "train": "../transformer/pfam_tiny_1579.train.csv",
        "valid": None,
        "test": None,
        "batch_size": 32,
        "num_epochs": 7,
        "lr": 1e-3,
        "emb_type": "ESM",
        "label": "label",
        # "sequence": "seq",
        "sequence": "original",
        "target": "family",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "experiment_name": "runs/esmSmall",
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
