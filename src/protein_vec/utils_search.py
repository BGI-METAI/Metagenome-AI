import numpy as np
import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import gc
import h5py
import torch
from collections import defaultdict
from torch import nn
from torch.utils.data import DataLoader
import faiss
import os



def load_database(lookup_database):
    #Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)

    return(index)


def query(index, queries, k=10):
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)

    return(D, I)

def featurize_prottrans(sequences, model, tokenizer, device): 
    
    sequences = [(" ".join(sequences[i])) for i in range(len(sequences))] #TODO 
    sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
    ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True) #make tokenization for all sequences, addpecial tokens, padding to the longest sequence
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)

    with torch.no_grad():
        embedding = model(input_ids=input_ids, attention_mask=attention_mask)

    # may should cut the aditional token from the data

    embedding = embedding.last_hidden_state.cpu().numpy()
    prottrans_embedding = torch.tensor(embedding)
    prottrans_embedding = prottrans_embedding.to(device) # padding adds one more caracter (insted od 512 it is 513)
    
    return(prottrans_embedding)


def embed_vec(prottrans_embedding, model_deep, masks, device):
    padding = torch.zeros(prottrans_embedding.shape[0:2]).type(torch.BoolTensor).to(device)
    out_seq = model_deep.make_matrix(prottrans_embedding, padding)
    vec_embedding = model_deep(out_seq, masks)
    return(vec_embedding.cpu().detach().numpy())

def encode(sequences, model_deep, model, tokenizer, masks, device):
    i = 0
    embed_all_sequences=[]
    while i < len(sequences):
        protrans_sequence = featurize_prottrans(sequences[i:i+1], model, tokenizer, device)
        embedded_sequence = embed_vec(protrans_sequence, model_deep, masks, device)
        embed_all_sequences.append(embedded_sequence)
        i = i + 1
    return np.concatenate(embed_all_sequences, axis=0)