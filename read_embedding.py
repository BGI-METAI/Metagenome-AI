import glob
import csv
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import pickle


# 每个文件夹下 只有自己的 .pkl文件 只需要获取当前文件夹下 所有pkl 文件信息
# 

class CustomDataset(Dataset):
    def __init__(self, embeddings_dir=None, emb_type=None):
        """
        embeddings_dir: str 为保存的 embedding目录
        emb_type: str 为 embedding的类型 有 mean max cls
        """
        self.sequences_path = embeddings_dir
        self.files = glob.glob(f"{embeddings_dir}/*.pkl")
        self.emb_type = emb_type
        with open(f"{self.embeddings_dir}/{self.files[0]}.pkl", "rb") as file_emb:
            self.length = len(pickle.load(file_emb))

    def __len__(self):
        total_length = 0
        for file_name in self.files:
            try:
                # 打开文件并加载字典
                with open(f"{self.embeddings_dir}/{file_name}.pkl", "rb") as file_emb:
                    emb_dict = pickle.load(file_emb)
                    total_length += len(emb_dict)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
        return total_length

    def __getitem__(self, idx):

        # length 应为 10240 为一个字典的切片长度
        file_id = idx // self.length
        slice_id = idx % self.length

        # 挨个读取存储 文件夹下的 字典 embedding
        with open(f"{self.embeddings_dir}/{self.files[file_id]}.pkl", "rb") as file_emb:
            prot_emb = pickle.load(file_emb)

        sample = prot_emb[slice_id]

        if sample["emb"] == "mean":
            emb = sample["emb"][0]
        elif sample["emb"] == "max":
            emb = sample["emb"][1]
        else:
            emb = sample["emb"][2]

        return {
            "protein_id": sample["protein_id"],
            "len": sample["len"],
            "sequence": sample["sequence"],
            "emb": emb,
            "labels": sample["label"],
        }
    
    def collate_fn(self, batch):
        """
        只需要按 batch_size 来合并 emb 和 lable为 张量形式 即可
        """

        # 初始化列表来收集 'emb' 和 'label'
        emb_list = []
        label_list = []

        # 遍历批次中的每个字典
        for item in batch:
            # 将 'emb' 和 'label' 添加到相应的列表
            emb_list.append(item['emb'])
            label_list.append(item['label'])

        # 将 'emb' 列表中的所有向量堆叠成一个二维张量
        emb_tensor = torch.stack(emb_list)

        # 将 'label' 列表转换为张量
        # 假设 'label' 是一个标量或一维列表，这里将其转换为一维张量
        label_tensor = torch.tensor(label_list, dtype=torch.long)

        return {'emb': emb_tensor, 'label': label_tensor}



