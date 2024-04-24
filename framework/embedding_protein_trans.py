import gc
import re
import logging

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from embedding import Embedding

# Adapted from https://github.com/tymor22/protein-vec/blob/main/src_run/gh_encode_and_search_new_proteins.ipynb

class ProteinTransEmbedding(Embedding):
    def __init__(self, model_name='prot_t5_xl_uniref50'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        #Load the ProtTrans model and ProtTrans tokenizer
        # https://huggingface.co/Rostlab
        # Available models in Rostlab:
        # prot_bert, ProstT5, ProstT5_fp16,prot_t5_xl_uniref50, prot_t5_xl_half_uniref50-enc,
        # prot_t5_base_mt_uniref50, prot_t5_base_mt_uniref50, prot_bert_bfd_ss3, prot_bert_bfd_membrane,
        # prot_bert_bfd_localization, prot_t5_xxl_uniref50
        self.tokenizer = T5Tokenizer.from_pretrained(f"Rostlab/{model_name}", do_lower_case=False )
        self.model = T5EncoderModel.from_pretrained(f"Rostlab/{model_name}")

        self.model.to(self.device)
        self.model.eval()

        logging.info(f'Number of parameters in {model_name} model: ', sum(p.numel() for p in  self.model.parameters()))

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
        features = []
        for seq_num in range(len(embedding)):  # TODO: Consider removing this if it is too slow!
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            features.append(seq_emd)

        prottrans_embedding = torch.tensor(features)
        prottrans_embedding = prottrans_embedding.to(device) # padding adds one more caracter (insted od 512 it is 513)
        
        return(prottrans_embedding)
