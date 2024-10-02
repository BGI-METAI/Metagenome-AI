import argparse
import datetime
import json
import logging
import os
import random
import re
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer
from transformers import AdamW, get_linear_schedule_with_warmup

import esm
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer

class T5EncoderWithLinearDecoder(torch.nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_uniref50'):
        super(T5EncoderWithLinearDecoder, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.encoder.config.d_model, self.encoder.config.vocab_size)
        # self.relu = torch.nn.ReLU() # TODO:

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        logits = self.linear(hidden_states)
        return logits

def finetune(config):
    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    logging.basicConfig(
        level=logging.WARNING,  # Set the log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format='%(asctime)s - %(levelname)s - %(message)s',  # Include date and time in the log
        datefmt='%Y-%m-%d %H:%M:%S',  # Set the format of the date and time
        filename=f"{config['model_type']}_{timestamp}.log",
    )

    user_home = str(Path.home())
    logging.info(user_home)
    # ESM2 and ESM3 have same library name (esm). Assuming user will pass adequate environment
    if config["model_type"] == 'ESM':
        esm_model_path = config.get("model_name_or_path", "esm2_t33_650M_UR50D")
        model, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_path)
    elif config["model_type"] == 'ESM3':
        model = ESM3_sm_open_v0("cuda")
        alphabet = EsmSequenceTokenizer()
    else:  # PTRANS
        model_path = config['model_name_or_path']
        alphabet = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True, legacy=False)
        model = T5EncoderWithLinearDecoder(model_path)
    logging.info(f"Number of parameters in {config['model_type']} model: {sum(p.numel() for p in model.parameters())}")

    data = list()
    # Combining train test and validation dataset to be used for finetuning model
    for key in ['train', 'test', 'valid']:
        if key in config and os.path.exists(config[key]):
            df = pd.read_csv(config[key], header=None, sep='\t')
            for ind, row in df.iterrows():
                if config["model_type"] == 'ESM':
                    data.append((row[0], row[2]))
                else:  # PTRANS
                    data.append(row[2])
    logging.info(f'Size of fine-tuning data: {len(data)}')

    # Check if a GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f'Finetuning model on: {device}')
    model = model.to(device)

    if config["model_type"] == 'ESM':
        batch_converter = alphabet.get_batch_converter()
        # Convert the data to batch format
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
    elif config["model_type"] == "ESM3":
        tokens = alphabet.batch_encode_plus(data, padding=True)[
            "input_ids"
        ]  # encode
        batch_tokens = torch.tensor(tokens, dtype=torch.int64)
    else:  # PTRANS
        # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        data = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data]
        batch_tokens = alphabet.batch_encode_plus(data, add_special_tokens=True, padding=True) #make tokenization for all sequences, addpecial tokens, padding to the longest sequence
        batch_tokens = torch.tensor(list(batch_tokens['input_ids']))

    mask_idx = alphabet.mask_idx  if config["model_type"] == 'ESM' else alphabet.unk_token_id
    pad_idx = alphabet.padding_idx if config["model_type"] == 'ESM' else alphabet.pad_token_id
    
    if config["model_type"] == 'ESM3':
        mask_idx = alphabet.mask_token_id

    # Prepare a dataset and dataloader for batching
    dataset = TensorDataset(batch_tokens)
    dataloader = DataLoader(dataset, batch_size=config["batch_size_finetune"], shuffle=True)

    # Define a loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # Setup optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=3e-5)
    total_training_steps = (len(data) // config["batch_size_finetune"]) * config["num_epochs_finetune"]
    num_warmup_steps = int(0.1 * total_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=num_warmup_steps,
                                            num_training_steps=total_training_steps)

    max_mask_prob = config["max_mask_prob"]

    # Masking function
    def mask_tokens(tokens, mask_idx, pad_idx, mask_prob=max_mask_prob):
        masked_tokens = tokens.clone()
        current_mask_prob = random.random() * max_mask_prob
        # Create a mask based on the probability
        mask = (torch.rand(tokens.shape) < current_mask_prob) & (tokens != pad_idx)
        # Replace masked positions with the mask index
        masked_tokens[mask] = mask_idx  # TODO: Replace with some other token instead MASK token or leave unchanged
        return masked_tokens

    # Enable training mode
    model.train()

    # Fine-tuning loop
    finetuned_output_file = ''
    for epoch in range(config["num_epochs_finetune"]):
        total_loss = 0
        for ind, batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            if (config["model_type"] in ['ESM', 'ESM3']):
                original_tokens = batch[0]
            else: #PTRANS
                original_tokens = batch[0]
                attention_mask = (original_tokens != pad_idx).long().to(device)

            # Mask tokens
            if (config["model_type"] in ['ESM', 'ESM3']): mask_idx = random.randint(4, 30) # All tokens can be seen using alphabet.tok_to_idx

            masked_tokens = mask_tokens(original_tokens, mask_idx, pad_idx)
            masked_tokens = masked_tokens.to(device)
            original_tokens = torch.tensor(original_tokens).to(device)  # Move original_tokens (labels) to GPU

            # Forward pass: get the output from the model
            # with torch.no_grad():
            if config["model_type"] == 'ESM':
                output = model(masked_tokens, repr_layers=[33])
                logits = output["logits"]
            elif config["model_type"] == 'ESM3':
                output = model(sequence_tokens=masked_tokens)
                logits = output.embeddings
            else:  # PTRANS
                logits = model(input_ids=masked_tokens, attention_mask=attention_mask)
            #  argmax on 33 size vector (size of vocabulary) is performed inside CrossEntropyLoss function
            loss = criterion(logits.view(-1, logits.size(-1)), original_tokens.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if ind % 50 == 0:
                logging.debug(f'Processing batch: {ind}, loss={loss.item()}, total_loss = {total_loss}')

        avg_loss = total_loss / len(data) 
        logging.info(f"Epoch {epoch+1}/{config['num_epochs_finetune']}, Loss: {avg_loss:.4f}")

        # Save the fine-tuned model
        finetuned_output_name = f"{config['model_type']}_finetuned_epoch_{epoch+1}.pt"
        if (config["model_type"] in ['ESM', 'ESM3']):
            torch.save(model.state_dict(), finetuned_output_name)
        else:  # PTRANS
            torch.save(model.encoder.state_dict(), finetuned_output_name)
        logging.info(f"Saved model name: {finetuned_output_name}")
    return os.path.join(user_home,finetuned_output_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        help="Determines the path to the config file.",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    with open(args.config_path) as file:
            config = json.load(file)
    logging.info(finetune(config))