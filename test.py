import argparse
import datetime
import logging
import warnings
import os
from pathlib import Path
import csv
import socket

from matplotlib import pyplot as plt
import datasets
import pandas as pd
import pickle
import pyarrow as pa
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import torch

# from torcheval.metrics import MultilabelAccuracy
from tqdm import tqdm
import wandb
import time

# Enable Multi-GPU training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# from embed.embedding_dataset import TSVDataset, MaxTokensLoader
from utils.config import get_weights_file_path, ConfigProviderFactory

from read_embedding import CustomDataset
from abc import abstractmethod, ABCMeta
from embed.embedding import Embedding
from embed.dataset import TSVDataset, CSVDataset

try:
    from embed.embedding_esm import EsmEmbedding
except ImportError:
    print("You are missing some of the libraries for ESM")
try:
    from embed.embedding_esm3 import Esm3Embedding
except ImportError:
    print("You are missing some of the libraries for ESM3")
try:
    from embed.embedding_protein_trans import ProteinTransEmbedding
except ImportError:
    print("You are missing some of the libraries for ProteinTrans")
try:
    from embed.embedding_protein_vec import ProteinVecEmbedding
except ImportError:
    print("You are missing some of the libraries for ProteinVec")


def choose_llm(config):
    """Select a pretrained model that produces embeddings

    Args:
        config (_type_): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        Embedding: A subclass of class Embedding
    """
    if config["emb_type"] == "ESM":
        return EsmEmbedding()
    if config["emb_type"] == "ESM3":
        return Esm3Embedding()
    elif config["emb_type"] == "PTRANS":
        if (
            "prot_trans_model_path" not in config.keys()
            or config["prot_trans_model_path"] is None
        ):
            return ProteinTransEmbedding(model_name=config["prot_trans_model_name"])
        else:
            return ProteinTransEmbedding(
                model_name=config["prot_trans_model_path"], read_from_files=True
            )
    elif config["emb_type"] == "PVEC":
        return ProteinVecEmbedding()
    else:
        raise NotImplementedError("This type of embedding is not supported")

 

def store_emb_file(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 直接加载数据集，不需要分布式数据加载
    # ds = CSVDataset(config["sequences_path"], config["emb_dir"], config["emb_type"])
    ds = CSVDataset(config["sequences_path"])
    dataloader = torch.utils.data.DataLoader(ds, batch_size=config["batch_size"], shuffle=False)
    
    # 每个chunk文件中包含的key值数量
    chunk_key_size = config["chunk_key_size"]

    start_time = time.time()
    embeddings_chunks = {}

    llm = choose_llm(config)
    llm.to(device)

    for batch in tqdm(dataloader):
        with torch.no_grad():
            # 假设 res 是从 llm.store_embeddings 获取的嵌入结果
            res, out_dir = llm.store_embeddings(batch, config["emb_dir"])
        # 如果output 中的 key值数量 超过 10240 就保存到本地
        embeddings_chunks.update(res)
        # 检查列表长度是否达到50
        if len(embeddings_chunks) == chunk_key_size:
            # 保存当前列表中的所有embeddings_dict到一个文件
            chunk_file_name = f"{out_dir}/chunk_{len(embeddings_chunks)//chunk_key_size}.pkl"
            with open(chunk_file_name, "wb") as file:
                pickle.dump([emb_dict for emb_dict in embeddings_chunks], file)
            
            # 清空列表以便收集下一批
            embeddings_chunks.clear()

    # 检查是否有剩余的embeddings_dict需要保存
    if embeddings_chunks:
        chunk_file_name = f"{out_dir}/chunk_{len(embeddings_chunks)//chunk_key_size + 1}.pkl"
        with open(chunk_file_name, "wb") as file:
            pickle.dump([emb_dict for emb_dict in embeddings_chunks], file)
    end_time = time.time()
    print(f"Elapsed time: {(end_time - start_time)/60} min")


if __name__ == '__main__':
    
    config_path = "/home/share/huadjyin/home/wangshengfu/89_esm3/configs/config_esm3.json"
    # 你的配置和 world_size 定义
    config = ConfigProviderFactory.get_config_provider(config_path)
    # 你的配置信息
    # world_size = torch.cuda.device_count()
    world_size = 1  # 根据你的硬件配置设置

    # 创建 store_embeddings 类的实例
    # 调用 store_emb_file 方法
    store_emb_file(config)
