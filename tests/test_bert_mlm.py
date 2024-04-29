import sklearn
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
import os
import math

from proteinNER.base_module.dataset import CustomMaskDataset

model_dir = "/home/share/huadjyin/home/s_sukui/02_data/01_model/"
model_name = "roberta-base"
train_file = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/train.txt"
eval_file = "/home/share/huadjyin/home/s_sukui/02_data/05_meta/sz_4d/03_datasets/test.txt"
max_seq_length = 512
out_model_path = os.path.join(model_dir, 'pretain')
train_epoches = 10
batch_size = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_MODE"] = "offline"

# 这里不是从零训练，而是在原有预训练的基础上增加数据进行预训练，因此不会从 config 导入模型
tokenizer = AutoTokenizer.from_pretrained(os.path.join(model_dir, model_name), use_fast=True)

model = AutoModelForMaskedLM.from_pretrained(os.path.join(model_dir, model_name))

# prepare dataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

eval_dataset = LineByLineTextDataset(
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
