import os
import re
import argparse
import torch
import esm
import json
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel


class T5EncoderWithLinearDecoder(torch.nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_uniref50'):
        super(T5EncoderWithLinearDecoder, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.encoder.config.d_model, self.encoder.config.vocab_size)
        # self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        logits = self.linear(hidden_states)
        # logits = self.relu(logits)  # Applying ReLU
        return logits

def finetuine(config):
    user_home = str(Path.home())
    print(user_home)
    
    if config["model_type"] == 'ESM':
        # Load the pre-trained ESM-2 model and its alphabet, this makes sure that model is in .cache
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        # model_path=f"{user_home}/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt"  # esm2_t33_650M_UR50D
        # contact_regression_model = torch.load(f"{user_home}/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt")
        # model_data = torch.load(str(model_path), map_location="cpu")
    else:  # PTRANS
        model_path = config['ptrans_model_name_or_path']
        alphabet = T5Tokenizer.from_pretrained(model_path, do_lower_case=False, local_files_only=True, legacy=False)
        model = T5EncoderWithLinearDecoder(model_path)
    print(f'Number of parameters in {model_path} model: ', sum(p.numel() for p in  model.parameters()))

    data = list()
    for key in ['train', 'test', 'valid']:
        if key in config and os.path.exists(config[key]):
            df = pd.read_csv(config[key], header=None, sep=config['separator_data_finetune'])
            for _, row in df.iterrows():
                if config["model_type"] == 'ESM':
                    data.append((row[0], row[2]))
                else:  # PTRANS
                    data.append(row[2])
    print(f'Size of fine-tuning data: {len(data)}')

    # Check if a GPU is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    # model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model = model.to(device)

    if config["model_type"] == 'ESM':
        batch_converter = alphabet.get_batch_converter()
        # Convert the data to batch format
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
    else:  # PTRANS
        # this will replace all rare/ambiguous amino acids by X and introduce white-space between all amino acids
        data = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in data]
        batch_tokens = alphabet.batch_encode_plus(data, add_special_tokens=True, padding=True) #make tokenization for all sequences, addpecial tokens, padding to the longest sequence
        # input_ids = torch.tensor(batch_tokens['input_ids']).to(device)
        # attention_mask = torch.tensor(batch_tokens['attention_mask']).to(device)
        # with torch.no_grad():
        #     embedding = model(input_ids=input_ids, attention_mask=attention_mask)
        # batch_tokens = torch.tensor(list(zip(batch_tokens['input_ids'], batch_tokens['attention_mask'])))
        batch_tokens = torch.tensor(list(batch_tokens['input_ids']))

    mask_idx = alphabet.mask_idx  if config["model_type"] == 'ESM' else alphabet.unk_token_id  # torch.tensor(alphabet.mask_idx).to(device)
    pad_idx = alphabet.padding_idx if config["model_type"] == 'ESM' else alphabet.pad_token_id  # torch.tensor(alphabet.padding_idx).to(device)

    # Prepare a dataset and dataloader for batching
    dataset = TensorDataset(batch_tokens)
    dataloader = DataLoader(dataset, batch_size=config["batch_size_finetune"], shuffle=True)

    # Define a loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    max_mask_prob = 0.20

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
        for batch in tqdm(dataloader):
            # batch_tokens = batch[0]
            optimizer.zero_grad()

            if config["model_type"] == 'ESM':
                original_tokens = batch[0]
            else:
                original_tokens = batch[0][0]
                attention_mask = (original_tokens != pad_idx).long().to(device)
                # attention_mask = torch.tensor(batch[0][0][1]).to(device)

            # Mask tokens
            masked_tokens = mask_tokens(original_tokens, mask_idx, pad_idx)
            masked_tokens = masked_tokens.to(device)
            original_tokens = torch.tensor(original_tokens).to(device)  # Move original_tokens (labels) to GPU

            # Forward pass: get the output from the model
            # with torch.no_grad():
            if config["model_type"] == 'ESM':
                output = model(masked_tokens, repr_layers=[33])
                logits = output["logits"]
            else:  # PTRANS
                # input_ids = torch.tensor(masked_tokens).to(device)
                # attention_mask = torch.tensor(batch['attention_mask']).to(device)
                logits = model(input_ids=masked_tokens.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0))
            #  argmax on 33 size vector (size of vocabulary) is performed inside CrossEntropyLoss function
            loss = criterion(logits.view(-1, logits.size(-1)), original_tokens.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if ind % 10 == 0:
                print(f'Processing batch: {ind}, loss={loss.item()}, total_loss = {total_loss}')

        avg_loss = total_loss / len(data) 
        print(f"Epoch {epoch+1}/{config['num_epochs_finetune']}, Loss: {avg_loss:.4f}")

        # Save the fine-tuned model
        if config["model_type"] == 'ESM':
            finetuned_output_file = f"esm2_{int(max_mask_prob*100)}_perc_masked_model_{epoch}.pt"
            torch.save(model.state_dict(), finetuned_output_file)
            # torch.save({"model": model.state_dict(), "args": model_data['args'], "cfg": model_data['cfg']}, finetuned_output_file)
            # Save contact regression model as well because esm.pretrained.load_model_and_alphabet
            # requires this file named the same like model with suffix -contact-regression.pt
            # torch.save(contact_regression_model, f"esm2_{int(max_mask_prob*100)}_perc_masked_model_{epoch}-contact-regression.pt")
        else:  # PTRANS
            finetuned_output_file = f"ptrans_finetuned_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), finetuned_output_file)
            # Load later with:
            # model = T5EncoderWithLinearDecoder(model_name)
            # model.load_state_dict(torch.load('custom_t5_encoder_with_linear_decoder.pth'))
    return finetuned_output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_path",
        help="Determines the path to the config file.",
        type=str,
        required=False,
        default="ESM",
    )          
     
    args = parser.parse_args()
    with open(args.config_path) as file:
            config = json.load(file)
    print(finetuine(config))