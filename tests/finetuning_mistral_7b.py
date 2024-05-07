# Ref: https://colab.research.google.com/drive/1OT5IDtGIkMDL68GUj8rWKb4OYyHs18KW#scrollTo=PjXNXEVDj0ui
import os
import sklearn
from copy import deepcopy
from random import randrange
from functools import partial

import torch
import bitsandbytes as bnb

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["WANDB_MODE"] = "offline"

# model_name = "microsoft/phi-2"
model_name = "/home/share/huadjyin/home/s_sukui/02_data/01_model/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# bits and bytes configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_use_double_quant=False,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    # bnb_4bit_quant_storage=torch.uint8 # 需要拉去transfomers的torch-dtype-itemsize分支来修复bug
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Auto selects device to put model on.
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


# LoRA
def find_all_linear_names(model):
    # https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
    cls = bnb.nn.Linear4bit  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # lm_head is often excluded.
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


modules = find_all_linear_names(model)
lora_alpha = 16
lora_dropout = 0.1
lora_r = 8

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=modules,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

data_path = "/home/share/huadjyin/home/s_sukui/02_data/06_datasets/Puffin"
dataset = load_dataset(data_path, split="train")
random_sample = dataset[randrange(len(dataset))]


def format_prompt(sample):
    """Given a sample dictionary with key "conversations", format the conversation into a prompt.

    Args:
      sample: A sample dictionary from a Hugging Face dataset.

    Returns:
      sample: sample dictionary with "text" key for the formatted prompt.
    """

    INTRO = "Below is a conversation between a user and you."
    END = "Instruction: Write a response appropriate to the conversation."

    conversations = ""
    for response in sample["conversations"]:
        from_, value = response["from"], response["value"]
        conversations += f"<{from_}>: " + value + "\n"

    sample["text"] = "\n\n".join([INTRO, conversations, END])

    return sample


def get_max_length(model):
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            print(f"Found max length: {max_length}")
            break
    if not max_length:
        max_length = 1024
        print(f"Using default max length: {max_length}")
    return max_length


# max_length = get_max_length(model)
max_length = 2048


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: str, seed: int = 42):
    # Format each prompt.
    print("Preprocessing dataset...")
    dataset = dataset.map(format_prompt)

    # https://blog.ovhcloud.com/fine-tuning-llama-2-models-using-a-single-gpu-qlora-and-ai-notebooks/
    def preprocess_batch(batch, tokenizer, max_length):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )

    # Apply preprocessing to each batch of the dataset & and remove "conversations" and "text" fields.
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["conversations", "text"],
    )

    # Filter out samples that have input_ids exceeding max_length.
    # Not needed as the tokenizer truncates all prompts over max length.
    # dataset = dataset.filter(lambda sample: len(sample["input_ids"]) < max_length)

    # Shuffle dataset.
    dataset = dataset.shuffle(seed=seed)

    return dataset


formatted_dataset = deepcopy(dataset).map(format_prompt)
dataset = preprocess_dataset(tokenizer, max_length, dataset)

# Training
training_args = TrainingArguments(
    output_dir="outputs",
    per_device_train_batch_size=1,
    # Best practice: https://huggingface.co/docs/transformers/main/main_classes/quantization#tips-and-best-practices
    gradient_accumulation_steps=4,  # Powers of 2.
    learning_rate=2e-4,
    max_grad_norm=1.0,
    max_steps=40,
    lr_scheduler_type="linear",
    warmup_steps=5,
    fp16=True,
    logging_strategy="steps",
    logging_steps=1,
    save_strategy="steps",
    save_steps=10,
    optim="paged_adamw_8bit",
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset,
)
results = trainer.train()
