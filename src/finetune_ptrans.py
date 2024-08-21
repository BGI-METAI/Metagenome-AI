import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5EncoderModel
import torch.nn.functional as F
import random
# from tqdm import tqdm
from pathlib import Path
import pandas as pd

class T5EncoderWithLinearDecoder(torch.nn.Module):
    def __init__(self, model_name='Rostlab/prot_t5_xl_uniref50'):
        super(T5EncoderWithLinearDecoder, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_name)

        # self.encoder = T5EncoderModel.from_pretrained(model_name)
        self.linear = torch.nn.Linear(self.encoder.config.d_model, self.encoder.config.vocab_size)
        # self.relu = torch.nn.ReLU()

    def forward(self, input_ids, attention_mask=None):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state
        logits = self.linear(hidden_states)
        # logits = self.relu(logits)  # Applying ReLU
        return logits

# Define the dataset class
class ProteinDataset(Dataset):
    def __init__(self, sequences, tokenizer, mask_prob):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        inputs = self.tokenizer(sequence, return_tensors="pt")
        # print(idx, sequence)
        labels = inputs.input_ids.clone()
        print('inputs', inputs)
        print('labels', labels)
        curr_mask_prob = random.uniform(0, self.mask_prob)
        
        # Randomly mask tokens
        rand = torch.rand(labels.shape)
        mask_arr = (rand < curr_mask_prob) * (labels != self.tokenizer.pad_token_id)
        labels[~mask_arr] = tokenizer.pad_token_id  # Only calculate loss on masked tokens
        
        # print('curr_mask_prob', curr_mask_prob)
        # print('rand', rand)
        # print('mask_arr', mask_arr)
        # print('rand', rand)
        # print('labels', labels)

        # print('inputs.input_ids', inputs.input_ids)

        # print(inputs.input_ids.squeeze(0), labels.squeeze(0))
        
        return inputs.input_ids.squeeze(0), labels.squeeze(0)



# Initialize tokenizer and model
# tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50')
# model = T5ForConditionalGeneration.from_pretrained('Rostlab/prot_t5_xl_uniref50')
ptrans_model_path = "/home/share/huadjyin/home/vladimir/Metagenome-AI/framework/prot_t5_xl_uniref50"
tokenizer = T5Tokenizer.from_pretrained(ptrans_model_path, do_lower_case=False, local_files_only=True, legacy=False)
model = T5EncoderWithLinearDecoder(ptrans_model_path)
print(f'Number of parameters in {ptrans_model_path} model: ', sum(p.numel() for p in  model.parameters()))

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Example sequences (replace with your own data)
sequences = [
    "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPDWQNYQPQKAPQVLPVEPEVF",
    "SNDGKTKAIRSVNAGTPFNPETDLRIVQPLGVSLFEGPIPTLEFVNNAKT"
]

sequences = list()
df_paths = ['/home/share/huadjyin/home/chenjunhong/META_AI/datasets/AMP/AMP_2024_08_09.tsv', 
'/home/share/huadjyin/home/chenjunhong/META_AI/datasets/mariana_trench/mariana_to_100.tsv']
for df_path,sep in zip(df_paths, ['\t', ' ']):
    df = pd.read_csv(df_path, header=None, sep=sep)
    for ind, row in df.iterrows():
        sequences.append(row[2])
        if ind > 500:
            break
    break

# Create dataset and dataloader
dataset = ProteinDataset(sequences, tokenizer, mask_prob=0.15)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Optimizer and Loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for ind, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=inputs)  # , labels=labels
        logits = outputs#['logits']
        
        # Reshape logits and labels for loss computation
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        
        # Compute loss
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if ind % 10 == 0:
            print(f'Processing batch: {ind}, loss={loss.item()}, total_loss = {total_loss}')
        # break

    avg_loss = total_loss / len(sequences) 
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save the model after each epoch
    # model.save_pretrained(f"ptrans_finetuned_epoch_{epoch+1}")
    torch.save(model.state_dict(), f"ptrans_finetuned_epoch_{epoch+1}.pt")
    # Load later with:
    # model = T5EncoderWithLinearDecoder(model_name)
    # model.load_state_dict(torch.load('custom_t5_encoder_with_linear_decoder.pth'))

    # Probably no need to save tokenizer but let's be sure
    # tokenizer.save_pretrained(f"ptrans_finetuned_epoch_{epoch+1}")