#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   utils.py
@Time    :   2024/05/12 21:36:34
@Author  :   Nikola Milicevic 
@Version :   1.0
@Contact :   nikola260896@gmail.com
@License :   (C)Copyright 2024, Nikola Milicevic
@Desc    :   Utility functions, mainly ones used for memory consumption debugging
"""
import torch
import subprocess as sp
import os


def torch_gpu_mem_info(device=None):
    if not device:
        device = "cuda:0"
    alloc = torch.cuda.memory_allocated(device) / (1024 * 1024)
    res = torch.cuda.memory_reserved(device) / (1024 * 1024)
    max_alloc = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    max_res = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
    print(
        f"Allocated: {alloc} Reserved: {res} Max_alloc: {max_alloc} Max_res: {max_res}"
    )


def reset_cache_and_peaks():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def count_model_size_mb(model):
    return sum(p.nelement() * p.element_size() for p in model.parameters()) / (
        1024 * 1024
    )


def count_parameters(model):
    return sum(p.nelement() for p in model.parameters())


def check_gpu_mem(device=None):
    if not device:
        device = "cuda:0"
    free, total = torch.cuda.mem_get_info(device)
    print(f"Free: {free / (1024*1024)} Total: {total / (1024*1024)}")


def get_tensor_size_mb(t):
    sz = t.element_size() * t.nelement()
    sz /= 1024 * 1024
    print(f"Size MiB: {sz}")


def is_model_on_gpu(model):
    print(f"Model is on GPU: {next(model.parameters()).is_cuda}")


def check_gpu_used_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_used_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    print("[GPUs]: ", memory_used_info)
