import torch
from torch.utils.data import Dataset



class CustomDataset(Dataset):
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.label = config["label"]
        self.sequence = config["sequence"]
        self.target = config["target"]
        self.max_seq_len = config["max_seq_len"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        seq_label_pair = self.dataset[idx]
        sample = {
            "sequence": seq_label_pair[self.sequence][:self.max_seq_len],
            self.target: seq_label_pair["le_" + self.label],
        }
        return sample
