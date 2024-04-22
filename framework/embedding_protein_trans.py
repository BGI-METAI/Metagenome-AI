import gc
import re

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from embedding import Embedding

# Adapted from https://github.com/tymor22/protein-vec/blob/main/src_run/gh_encode_and_search_new_proteins.ipynb


class ProteinTransEmbedding(Embedding):
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Load the ProtTrans model and ProtTrans tokenizer
        # https://huggingface.co/Rostlab
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        gc.collect()

        self.model.to(self.device)
        self.model.eval()

        print('Number of parameters in ProTrans model: ', sum(p.numel() for p in  self.model.parameters()))

    def get_embedding(self, batch):
        
        #Pull out sequences for the new proteins
        flat_seqs = batch["sequence"] 
        protrans_embedings = self.featurize_prottrans(flat_seqs, self.model, self.tokenizer, self.device) #firt make embading using ProTrans pretrained model
        # if you want to derive a single representation (per-protein embedding) for the whole protein
        protrans_embedings = protrans_embedings.mean(dim=1) # shape (batch_size x 1024)

        return protrans_embedings  

    def get_embedding_dim(self):
        return self.model.shared.embedding_dim

    def to(self, device):
        self.model.to(device)

    def featurize_prottrans(self, sequences, model, tokenizer, device): 
        sequences = [(" ".join(sequences[i])) for i in range(len(sequences))] #TODO 
        sequences = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences]
        ids = tokenizer.batch_encode_plus(sequences, add_special_tokens=True, padding=True) #make tokenization for all sequences, addpecial tokens, padding to the longest sequence
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids, attention_mask=attention_mask)

        embedding = embedding.last_hidden_state.cpu().numpy()
        prottrans_embedding = torch.tensor(embedding)
        prottrans_embedding = prottrans_embedding.to(device) # padding adds one more caracter (insted od 512 it is 513)
        
        return(prottrans_embedding)
