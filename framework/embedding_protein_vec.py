import gc

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from protein_vec import trans_basic_block, trans_basic_block_Config
from protein_vec import featurize_prottrans, embed_vec
from embedding import Embedding

# Adapted from https://github.com/tymor22/protein-vec/blob/main/src_run/gh_encode_and_search_new_proteins.ipynb


class ProteinVecEmbedding(Embedding):
    def __init__(self):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Protein-Vec MOE model checkpoint and config
        # wget https://users.flatironinstitute.org/thamamsy/public_www/protein_vec_models.gz
        # # Unzip this directory of models with the following command:
        # tar -zxvf protein_vec_models.gz
        # Or download from /goofys/projects/MAI/protein_vec/
        vec_model_cpnt = 'Metagenome-AI/data/protein_vec/protein_vec.ckpt'  # ~800MB
        vec_model_config = 'Metagenome-AI/data/protein_vec/protein_vec_params.json'

        #Load the ProtTrans model and ProtTrans tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
        self.model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        gc.collect()

        self.model.to(self.device)
        self.model.eval()

        #Load the model
        vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
        self.model_deep = trans_basic_block.load_from_checkpoint(vec_model_cpnt, config=vec_model_config)
        self.model_deep = self.model_deep.to(self.device)
        self.model_deep = self.model_deep.eval()

        #self.batch_converter = self.tokenizer.get_batch_converter()  # TODO: Check if tokenizer has this method!

    def get_embedding(self, batch, pooling='cls'):
        #data = [(fam, seq) for fam, seq in zip(batch["family"], batch["sequence"])]
        #_, _, batch_tokens = self.batch_converter(data)

        # This is a forward pass of the Protein-Vec model
        # Every aspect is turned on (therefore no masks)
        sampled_keys = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]
        
        #Pull out sequences for the new proteins
        flat_seqs = batch["sequence"]   #ovde puca zato sto mu ne odgovara batch izgkeda da nije data frame

        #Loop through the sequences and embed them using protein-vec
        i = 0
        embed_all_sequences_in_batch = []
        while i < len(flat_seqs): 
            protrans_sequence = featurize_prottrans(flat_seqs[i:i+1], self.model, self.tokenizer, self.device) #firt make embading using ProTrans pretrained model
            embedded_sequence = embed_vec(protrans_sequence, self.model_deep, masks, self.device) #than use protrens embedings to get embedings from protein-vec model
            embed_all_sequences_in_batch.append(embedded_sequence)

        # with torch.no_grad():
        #     protrans_sequence = featurize_prottrans(data, self.model, self.tokenizer, self.device)
        #     embedded_sequence = embed_vec(protrans_sequence, self.model_deep, masks, self.device)

        return embed_all_sequences_in_batch  # TODO: Check if format of this embeddings is aligned with classifier layers


    def get_embedding_dim(self):
        return 512 # TODO: Fix

    def to(self, device):
        self.model_deep.to(device)
        self.model.to(device)
