import pickle
import random
from typing import Dict, List

import sklearn
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, PreTrainedTokenizer
from transformers import Trainer, TrainingArguments
import os
import math

# C:\Users\sukui\AppData\Local\JetBrains\PyCharm2021.3\remote_sources\-610422397\-1995996496\transformers\models\roberta\modeling_roberta.py

model_dir = "/home/share/huadjyin/home/s_sukui/02_data/01_model/"
model_name = "roberta-base"
train_file = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/train.txt"
eval_file = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/test.txt"
max_seq_length = 512
out_model_path = os.path.join(model_dir, 'pretain')
train_epoches = 10
batch_size = 10
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_MODE"] = "offline"
random.seed(42)

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), use_fast=True)

model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_dir, model_name))


class CustomDataset(Dataset):
    """
        This will be superseded by a framework-agnostic approach soon.
        """

    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 file_path: str,
                 block_size: int = 256,
                 ):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.pairs_path = []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file.readlines():
                self.pairs_path.extend([line.strip() for line in line.strip().split(' ') if line.endswith("pkl")])

    def __len__(self):
        return len(self.pairs_path)

    def __getitem__(self, idx) -> Dict[str, torch.tensor]:
        with open(self.pairs_path[idx], 'rb') as file:
            data = pickle.load(file)
        tokens = self.tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[" ".join(data['seq'])],
            padding='longest',
            max_length=self.block_size
        )
        input_ids = torch.tensor(tokens['input_ids'])
        return {'input_ids': torch.tensor(input_ids[0][:self.block_size], dtype=torch.long)}


# prepare dataset
dataset = CustomDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = CustomDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=128,
)
training_args = TrainingArguments(
    output_dir=out_model_path,
    overwrite_output_dir=True,
    num_train_epochs=train_epoches,
    per_device_train_batch_size=batch_size,
    save_steps=2000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model(out_model_path)
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
