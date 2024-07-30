#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   time_profiling.py
@Time    :   2024/04/17 09:19:05
@Author  :   Nikola Milicevic
@Version :   1.0
@Contact :   nikolamilicevic@genomics.cn
@Desc    :   None
"""

import esm
import random
import torch
from torch.nn import Identity
import numpy as np
import time
import subprocess as sp
import os
import pandas as pd

rand_seq = "RVFNPQLDRDGWQSTHTITIQVVTQDMPFLVDSIHMEINRLGLTTHLMIHIGGIKVCRNKKNQVDDVLAYHVQHHKESTLEAPISMEIDRQTDPKVLADIQRNIRRVLRDVRVAVEDWGLMRERVQEALSELDPAKMVQDPEQIKETKAFLNWLMDNHFTFLGFRDYELVGEGKEQALRLIPESGLGVLHDHTHSKMLRQYADLPKAARKMALSTEQILILSKTNTLSTVHRPAYTDYIGVKRFNEKGELIGERRFIGLYTSDVYRSDPRVIPIIRHKVESVLKRSQLPAKSHSGKDLLHILATLPSDDLFHATVDELFHWTMGILHLQERRRIRLFVRKDAYGRFMSCLVYVPRDYFTTDLVMRMQDILMKAFHGLDVSFTTYFSESILARIHFVIRINPRRALEYDVKELEEKLAKVGVSWEDEFYKHALDYFGEERGNDIFNRYRHAFSSAYREEFQAQQAVYDVAHIEKLSERTQLGMSIYRPRGAARDVIRFKLFHPDFTVPLSDALPMLENMGLRVVGEQPYELTFQDGRKVWINDFLMTYAREPEFEIETVKTIFQEAYEKIWFGAAEDDGLNRLVLEAQLTWREIAVFRAYMKYFRQVGFTFSEGYITDALVDNPKVARLLIELFKCYFDPERATTSKEKAQDIEQIIQKGLDEVAGLDEDRILRRYLALIHATLRTNYFQRDEKRNPKPYLSFKLDSSKIPDMPLPLPKYEIFVYSPRFEGVHLRGAAVARGGIRWSDRREDYRTEVLGLMKAQQVKNAVIVPAGAKGGFFPKRLPSEGSREEILQEGLFCYRNFIRGLLDLTDNLENGEIVSPKNTVCYDGPDPYLVVAADKGTATFSDVANSIAIEKNYWMGDAFASGGSTGYDHKKMGITARGAWVAAKRHFQDLGTNLDEAEITVVGIGDMSGDVFGNGMLISRYIKLVAAFDHRHIFLDPNPVPALSYEERLRLFNLPRSSWNDYDRSLLSAGGGVYSRAAKSIQLSPEVKALLHS"
device = torch.device("cuda:0")


def get_random_seq():
    return "".join(random.sample(rand_seq, len(rand_seq)))


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.used --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
# model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.contact_head = Identity()
model.emb_layer_norm_after = Identity()
model.lm_head = Identity()

# print("LEN", len(get_random_seq()))
# data
batch_converter = alphabet.get_batch_converter()

# data = [("protein2", get_random_seq()) for _ in range(80)]
# batch_labels, batch_strs, batch_tokens = batch_converter(data)

print(f"USED MEM BEFORE COPYING MODEL AND TENSORS: {get_gpu_memory()}")
model.to(device)
print(
    f"USED MEM AFTER COPYING MODEL TO GPU AND BEFORE COPYING TENSORS: {get_gpu_memory()}"
)

model.eval()

for i in range(5):
    print("-----------------------------")
    # num = random.choice([30, 20, 70, 80])
    data = [("protein2", get_random_seq()) for _ in range(80)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # print(f"Id of batch_tokens before {id(batch_tokens)}")
    batch_tokens = batch_tokens.to(device)
    # print(f"Id of batch_tokens {id(batch_tokens)}")
    print(f"USED MEM AFTER COPYING INPUT TENSORS TO GPU: {get_gpu_memory()}")
    with torch.no_grad():
        start = time.time()
        print(f"USED MEM BEFORE ESM RUN: {get_gpu_memory()}")
        res = model(batch_tokens)
        print(f"USED MEM AFTER ESM RUN: {get_gpu_memory()}")
        end = time.time()
        # print(f"ESM took {(end - start):.3} s")

print(f"USED MEM BEFORE PROGRAM END: {get_gpu_memory()}")
print(batch_tokens.shape)
