import gc

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from protein_vec import trans_basic_block, trans_basic_block_Config
from protein_vec import featurize_prottrans, embed_vec
from embeddings.embedding_protein_trans import ProteinTransEmbedding

# Adapted from https://github.com/tymor22/protein-vec/blob/main/src_run/gh_encode_and_search_new_proteins.ipynb


class ProteinVecEmbedding(ProteinTransEmbedding):
    def __init__(self, pvec_models='PFAM', ptrans_model_name='prot_t5_xl_uniref50'):
        super().__init__(ptrans_model_name)
        self.pvec_models = pvec_models
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #Protein-Vec MOE model checkpoint and config
        # wget https://users.flatironinstitute.org/thamamsy/public_www/protein_vec_models.gz
        # # Unzip this directory of models with the following command:
        # tar -zxvf protein_vec_models.gz
        # Or download from /goofys/projects/MAI/protein_vec/
        vec_model_cpnt = 'Metagenome-AI/data/protein_vec/protein_vec.ckpt'  # ~800MB
        vec_model_config = 'Metagenome-AI/data/protein_vec/protein_vec_params.json'

        #Load the model
        vec_model_config = trans_basic_block_Config.from_json(vec_model_config)
        self.model_deep = trans_basic_block.load_from_checkpoint(vec_model_cpnt, config=vec_model_config)
        self.model_deep = self.model_deep.to(self.device)
        self.model_deep = self.model_deep.eval()

        print('Number of parameters in ProTrans model: ', sum(p.numel() for p in  self.model.parameters()))
        print('Number of parameters in ProtVec model: ', sum(p.numel() for p in  self.model_deep.parameters()))

        # Every aspect is turned on (therefore no masks)
        #sampled_keys = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        sampled_keys = self.pvec_models.split(',')  #have to define the annotation that is in usage
        all_cols = np.array(['TM', 'PFAM', 'GENE3D', 'ENZYME', 'MFO', 'BPO', 'CCO'])
        self.masks = [all_cols[k] in sampled_keys for k in range(len(all_cols))]


    def get_embedding(self, batch):
        # This is a forward pass of the Protein-Vec model
        masks = np.tile(self.masks, (len(batch["sequence"]), 1)) # the size of the mask must be batch_size x 7 (number od aspect models)
        masks = torch.logical_not(torch.tensor(masks, dtype=torch.bool))[None,:]
        masks.squeeze_(0)
        
        #Pull out sequences for the new proteins
        flat_seqs = batch["sequence"] 

        protrans_sequence = super().featurize_prottrans(flat_seqs, self.model, self.tokenizer, self.device) #first make embading using ProTrans pretrained model
        embed_all_sequences_in_batch = embed_vec(protrans_sequence, self.model_deep, masks, self.device) #than use protrens embedings to get embedings from protein-vec model  

        embed_all_sequences_in_batch = torch.Tensor(embed_all_sequences_in_batch) #convert from list to tensor
        embed_all_sequences_in_batch = embed_all_sequences_in_batch.to(self.device) #radi i bez ovoga pitanje je zasto?
        return embed_all_sequences_in_batch


    def get_embedding_dim(self):
        return self.model_deep.config.out_dim

    def to(self, device):
        self.model_deep.to(device)
        super().to(device)
