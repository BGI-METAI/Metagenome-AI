import torch
import esm
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path

user_home = str(Path.home())
print(user_home)

# Load the pre-trained ESM-2 model and its alphabet, this makes sure that model is in .cache
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()

esm_model_path=f"{user_home}/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt"  # esm2_t33_650M_UR50D
contact_regression_model = torch.load(f"{user_home}/.cache/torch/hub/checkpoints/esm2_t33_650M_UR50D-contact-regression.pt")
model_data = torch.load(str(esm_model_path), map_location="cpu")

data = list()
df_paths = ['/home/share/huadjyin/home/chenjunhong/META_AI/datasets/AMP/AMP_2024_08_09.tsv', 
'/home/share/huadjyin/home/chenjunhong/META_AI/datasets/mariana_trench/mariana_to_100.tsv']
for df_path,sep in zip(df_paths, ['\t', ' ']):
    df = pd.read_csv(df_path, header=None, sep=sep)
    for ind, row in df.iterrows():
        data.append( (row[0], row[2]) )
    break
# data = [
#     ("protein1", "MKTAYIAKQRQISFVKSHFSRQDILDLWIYHTQGYFPQYKGSGRTQY"),
#     ("protein2", "GIEVVVNATLDKAGFQAGYIGFLKTFTLGVAGSGLLGGTYTQAGG"),
#     # Add more sequences here
# ]
print(f'Size of fine-tuning data: {len(data)}')

# Check if a GPU is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()

# Convert the data to batch format
batch_labels, batch_strs, batch_tokens = batch_converter(data)

# Prepare a dataset and dataloader for batching
dataset = TensorDataset(batch_tokens)
batch_size = 16
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=alphabet.padding_idx)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
max_mask_prob = 0.20

# Masking function
def mask_tokens(tokens, mask_idx, pad_idx, mask_prob=max_mask_prob):
    labels = tokens.clone()
    masked_tokens = tokens.clone()
    current_mask_prob = random.random() * max_mask_prob

    # Create a mask based on the probability
    mask = (torch.rand(tokens.shape) < current_mask_prob) & (tokens != pad_idx)

    # Replace masked positions with the mask index
    masked_tokens[mask] = mask_idx

    return masked_tokens, labels

# Enable training mode
model.train()

# Fine-tuning loop
num_epochs = 60
mask_idx = alphabet.mask_idx# torch.tensor(alphabet.mask_idx).to(device)
pad_idx = alphabet.padding_idx #torch.tensor(alphabet.padding_idx).to(device)

for epoch in range(num_epochs):
    for batch in tqdm(dataloader):
        batch_tokens = batch[0]
        optimizer.zero_grad()

        # Mask tokens
        masked_tokens, labels = mask_tokens(batch_tokens, mask_idx, pad_idx)
        masked_tokens = masked_tokens.to(device)
        labels = labels.to(device)  # Move labels to GPU

        # Forward pass: get the output from the model
        # with torch.no_grad():
        output = model(masked_tokens, repr_layers=[33])
        logits = output["logits"]
        # Take the argmax of the logits to get the predicted amino acids
        # predictions = torch.argmax(logits, dim=-1)

        # print(logits.size(), labels.size())
        # print(logits.view(-1, logits.size(-1)).size(), labels.view(-1).size())
        # Compute loss for masked language modeling
        # argmax on 33 size vector (size of vocabulary) is performed inside CrossEntropyLoss function
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item()}")

    # Save the fine-tuned model
    # torch.save(model.state_dict(), f"fine_tuned_esm2_20perc_masked_model_{epoch}.pth")
    # torch.save({"model": model.state_dict(), "args":{}, "cfg":{"epoch": epoch, "base_model": "esm2_t33_650M_UR50D", "max_mask_probability": max_mask_prob} },
    # f"esm2_{int(max_mask_prob*100)}_perc_masked_model_{epoch}.pt"
    torch.save({"model": model.state_dict(), "args": model_data['args'], "cfg": model_data['cfg']},
    f"esm2_{int(max_mask_prob*100)}_perc_masked_model_{epoch}.pt")

    # Save contact regression model as well because esm.pretrained.load_model_and_alphabet
    # requires this file named the same like model with suffix -contact-regression.pt
    torch.save(contact_regression_model, f"esm2_{int(max_mask_prob*100)}_perc_masked_model_{epoch}-contact-regression.pt")


