from pathlib import Path


def get_config():
    return {
        "train":
        "/home/share/huadjyin/home/nikolamilicevic/pfam_subs/pfam.train.subs.csv",
        "valid":
        "/home/share/huadjyin/home/nikolamilicevic/pfam_subs/pfam.valid.subs.csv",
        "test": "/home/share/huadjyin/home/nikolamilicevic/pfam_subs/pfam.test.subs.csv",
        "batch_size": 64,
        "num_epochs": 10,
        "lr": 1e-3,
        "emb_type": "ESM",
        "label": "label",
        "max_seq_len": 600 * 2,
        "sequence": "seq",
        "target": "family",
        "model_folder": "weights",
        "model_basename": "esm_model_",
        "preload": None,
        "experiment_name": "runs/esm",
        "tensorboard": False,
    }


def get_weights_file_path(config, epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path(".") / model_folder / model_filename)
