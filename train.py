import argparse
import datetime
import logging
import warnings
import wandb
import os
import socket

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from config import ConfigProviderFactory
from model import ComplexBinaryClassifier

class EmbeddingDataset(Dataset):
    def __init__(self, data, labels, max_length=None, device=None):
        self.data = data
        self.labels = labels
        self.max_length = max_length if max_length is not None else max(len(d[0]) for d in data)
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding, label = self.data[idx], self.labels[idx]
        
        # 将标签转换为tensor，并移动到与embedding相同的设备
        label = torch.tensor(label, dtype=torch.float).to(self.device).clone().detach()

        if embedding.size(1) < self.max_length:
            padding = torch.zeros(1, self.max_length - embedding.size(1), 1536, device=self.device)
            embedding = torch.cat((embedding, padding), 1)

        embedding = embedding.to(self.device)

        return embedding, label

def init_wandb(model_folder, timestamp, model=None):
    # initialize wandb tracker
    wandb_output_dir = os.path.join(model_folder, "wandb_home")
    Path(wandb_output_dir).mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project="protein function annotation",
        notes=socket.gethostname(),
        name=f"prot_func_anno_{timestamp}",
        group="linear_classifier",
        dir=wandb_output_dir,
        job_type="training",
        reinit=True,
    )
    if model:
        wandb.watch(model, log="all")

    return run

def init_logger(timestamp):
    logging.basicConfig(
        format="%(name)-12s %(levelname)-8s %(message)s",
        level=logging.INFO,
        filename=f"{timestamp}.log",
    )
    return logging.getLogger(__name__)

def train(model, dataloader, optimizer, criterion):
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.view(-1, 1))  # 将targets转为二维
        loss.backward()
        optimizer.step()

        # 记录训练过程中的指标
        wandb.log({
            "train_loss": loss,
        })

    print(f"Loss: {loss.item()}")

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        tp, fp, fn, tn = 0, 0, 0, 0
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predicted = (outputs > 0).float()  # 将输出转换为0或1

            tp += torch.sum((targets == 1) & (predicted.squeeze(1)== 1)).item() 
            tn += torch.sum((targets == 0) & (predicted.squeeze(1)== 0  )).item() 
            fp += torch.sum((targets == 0) & (predicted.squeeze(1)== 1)).item() 
            fn += torch.sum((targets == 1) & (predicted.squeeze(1)== 0)).item() 

    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")


def main(file_path):

    if os.path.exists(file_path):
        print("embedding文件存在，直接读取")
        label = torch.load("/home/share/huadjyin/home/wangshengfu/06_esm/data-lab-1w.pt")
        data = torch.load("/home/share/huadjyin/home/wangshengfu/06_esm/data-seq-1w.pt")
        # label = torch.load("/home/share/huadjyin/home/wangshengfu/06_esm/data-lab-all.pt")
        # data = torch.load("/home/share/huadjyin/home/wangshengfu/06_esm/data-seq-all.pt")
        split = int(len(data) * 0.8)
        data_train, label_train = data[:split], label[:split]
        data_val, label_val= data[split:], label[split:]
    else:
        print("embedding文件不存在，需要重新处理")

    # 假设data和label已经准备好，并且是列表形式
    # 然后在创建数据集时，指定max_length
    max_length = max(len(d[0]) for d in data)  # 找到最长的序列长度
    print(f"最长序列长度为{max_length}")
    # 在创建数据集时，指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试集
    dataset_val = EmbeddingDataset(data_val, label_val, max_length=max_length,device=device)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=512, shuffle=True)
    # model = BinaryClassifier().to(device)
    model = ComplexBinaryClassifier().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # 训练
    for epoc in range(10):
        # 训练集
        dataset_train = EmbeddingDataset(data_train, label_train, max_length=max_length,device=device)
        dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=512, shuffle=True)
        train(model, dataloader_train, optimizer, criterion)

    torch.save(model, 'esm3-amp-3linear.pth')

    model_val = torch.load('esm3-amp-3linear.pth')
    print(model_val)
    # 评估
    evaluate(model_val, dataloader_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Module that contains training and evaluation. Will be separated."
    )
    parser.add_argument(
        "-c",
        "--config_path",
        help="Type of embedding to be used",
        type=str,
        required=False,
        default="ESM",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    config = ConfigProviderFactory.get_config_provider(args.config_path)
    wandb.login(key=config["wandb_key"])
    world_size = torch.cuda.device_count()

    timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    logger = init_logger(timestamp)


    main()