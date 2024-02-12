import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_label_pair = self.dataset[idx]
        sample = {
            "sequence": seq_label_pair["original"],
            "family": seq_label_pair["le_label"],
        }
        return sample
