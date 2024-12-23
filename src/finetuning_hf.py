"""
The finetuning.huggingface_ft module uses Trainer API of 
huggingface which integrates accelerate.
With accelerate you can set up multi-GPU training.
In order to use accelerate, run:

> accelerate config

When the you enter the requested info and config gets created, 
launch the finetuning with:

> accelerate launch finetuning_hf.py
"""

import argparse
import datetime

import torch
import wandb

from config import ConfigProviderFactory
from finetuning.huggingface_ft import finetune

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script that performs finetuning of huggingface models."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
    )

    args = parser.parse_args()
    config = ConfigProviderFactory.get_config_provider(args.config_path)
    wandb.login(key=config["wandb_key"])
    world_size = torch.cuda.device_count()
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    print("Succesfully started the script for finetuning...")
    finetune(config)
