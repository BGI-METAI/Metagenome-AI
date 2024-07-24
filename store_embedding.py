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

from embed.embedding_dataset import TSVDataset, MaxTokensLoader
from utils.config import get_weights_file_path, ConfigProviderFactory

from read_embedding import CustomDataset
from abc import abstractmethod, ABCMeta
from embed.embedding import Embedding

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

class store_embeddings(Embedding):
    @abstractmethod
    def get_embedding(self):
        pass

    @abstractmethod
    def get_embedding_dim(self):
        pass

    @abstractmethod
    def to(self, device):
        pass

    def choose_llm(self, config):
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

    # Initialize the PyTorch distributed backend
    # This is essential even for single machine, multiple GPU training
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0)
    # The distributed process group contains all the processes that can communicate and synchronize with each other.
    def ddp_setup(self, rank, world_size):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        # IP address that runs rank=0 process
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def store_emb_file(self, rank, config, world_size):
        try:
            self.ddp_setup(rank, world_size)
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

            # 读取序列文件 生成数据集
            ds = TSVDataset(config["sequences_path"], config["emb_dir"], config["emb_type"])


            max_tokens = config["max_tokens"]
            chunk_size = len(ds) // world_size
            remainder = len(ds) % world_size

            # 每个chunk文件中包含的key值数量
            chunk_key_size = config["chunk_key_size"]

            start = 0
            indices = []
            for i in range(world_size):
                end = start + chunk_size + (1 if i < remainder else 0)
                indices.append((start, end))
                start = end

            # Each GPU gets its own part of the dataset
            dataloader = MaxTokensLoader(
                ds, max_tokens, start_ind=indices[rank][0], end_ind=indices[rank][1]
            )

            llm = self.choose_llm(config)
            llm.to(device)
            

            dist.barrier()

            start_time = time.time()
            embeddings_chunks = {}
            for batch in dataloader:
                try:
                    with torch.no_grad():
                        res, out_dir = llm.store_embeddings(batch, config["emb_dir"])
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print("Cuda was out of memory, recovering...")
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
        finally:
            destroy_process_group()

class store_emb_child(store_embeddings):
    def get_embedding(self):
        # 具体实现
        pass

    def get_embedding_dim(self):
        # 具体实现
        pass

    def to(self, device):
        # 具体实现
        print(f"Model moved to device {device}")

    def choose_llm(self, config):
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

    # Initialize the PyTorch distributed backend
    # This is essential even for single machine, multiple GPU training
    # dist.init_process_group(backend='nccl', init_method='tcp://localhost:FREE_PORT', world_size=1, rank=0)
    # The distributed process group contains all the processes that can communicate and synchronize with each other.
    def ddp_setup(self, rank, world_size):
        """
        Args:
            rank: Unique identifier of each process
            world_size: Total number of processes
        """
        # IP address that runs rank=0 process
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12356"
        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)

    def store_emb_file(self, rank, config, world_size):
        try:
            self.ddp_setup(rank, world_size)
            device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
            Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)

            # 读取序列文件 生成数据集
            ds = TSVDataset(config["sequences_path"], config["emb_dir"], config["emb_type"])


            max_tokens = config["max_tokens"]
            chunk_size = len(ds) // world_size
            remainder = len(ds) % world_size

            # 每个chunk文件中包含的key值数量
            chunk_key_size = config["chunk_key_size"]

            start = 0
            indices = []
            for i in range(world_size):
                end = start + chunk_size + (1 if i < remainder else 0)
                indices.append((start, end))
                start = end

            # Each GPU gets its own part of the dataset
            dataloader = MaxTokensLoader(
                ds, max_tokens, start_ind=indices[rank][0], end_ind=indices[rank][1]
            )

            llm = self.choose_llm(config)
            llm.to(device)
            

            dist.barrier()

            start_time = time.time()
            embeddings_chunks = {}
            for batch in dataloader:
                try:
                    with torch.no_grad():
                        res, out_dir = llm.store_embeddings(batch, config["emb_dir"])
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        print("Cuda was out of memory, recovering...")
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
        finally:
            destroy_process_group()

def run_store_embeddings(rank, config, world_size):

    # 创建 store_embeddings 类的实例
    embedding_manager = store_emb_child()
    # 调用 store_emb_file 方法
    embedding_manager.store_emb_file(rank, config, world_size)

if __name__ == '__main__':
    
    config_path = "/home/share/huadjyin/home/wangshengfu/89_esm3/configs/config_esm3.json"
    # 你的配置和 world_size 定义
    config = ConfigProviderFactory.get_config_provider(config_path)
    # 你的配置信息
    # world_size = torch.cuda.device_count()
    world_size = 1  # 根据你的硬件配置设置


    # 设置函数属性
    # run_store_embeddings.config = config
    # run_store_embeddings.world_size =  world_size

    # 使用 spawn 启动多进程
    mp.spawn(run_store_embeddings, args=(config, world_size), nprocs=world_size)
