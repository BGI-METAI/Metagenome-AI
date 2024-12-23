"""Script for finetuning of huggingface models.
"""

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import os


class CUDAMemoryLogger(TrainerCallback):
    def __init__(self, log_every_n_steps=100):
        self.log_every_n_steps = log_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        # Log CUDA memory usage every 'log_every_n_steps' steps
        if state.global_step % self.log_every_n_steps == 0:
            allocated_memory = torch.cuda.memory_allocated() / 1024**2  # in MB
            reserved_memory = torch.cuda.memory_reserved() / 1024**2  # in MB
            print(f"Step {state.global_step}:")
            print(f"  Allocated memory: {allocated_memory:.2f} MB")
            print(f"  Reserved memory: {reserved_memory:.2f} MB")

        return control


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    try:
        auc = roc_auc_score(labels, predictions)
    except ValueError:
        auc = "N/A"  # Handle the case when there are no positive labels
    mcc = matthews_corrcoef(labels, predictions)

    metrics = {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "AUC": auc,
        "MCC": mcc,
    }
    return metrics


def finetune(config):
    print(
        f"Function finetune running on GPU: {torch.cuda.current_device()} \nLOCAL_RANK: {os.environ.get('LOCAL_RANK')} \nVisible GPUs: {torch.cuda.device_count()}"
    )
    dataset = load_dataset(
        "csv",
        data_files={
            "train": config["train"],
            "validation": config["valid"],
            "test": config["test"],
        },
        delimiter="\t",
        header=None,
        column_names=["protein_id", "seq_len", "sequence", "label"],
    )

    checkpoint = config["checkpoint"]
    # checkpoint = ".cache/huggingface/hub/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc/"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["sequence"]
        )  # leave out padding=True, pad dynamically when batches are created

    tokenized_dataset = dataset.map(
        tokenize_function, batched=True
    )  # batched -> speedup tokenization

    keep_columns = ["input_ids", "attention_mask", "label"]

    for split in tokenized_dataset:
        tokenized_dataset[split] = tokenized_dataset[split].remove_columns(
            [
                col
                for col in tokenized_dataset[split].column_names
                if col not in keep_columns
            ]
        )

    for split in tokenized_dataset:
        # Rename the 'label' column to 'labels'
        tokenized_dataset[split] = tokenized_dataset[split].rename_column(
            "label", "labels"
        )
    tokenized_dataset

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=2, problem_type="single_label_classification"
    )

    if config["lora"]:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=128,
            lora_alpha=128,
            lora_dropout=0.05,
            target_modules="all-linear",
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    batch_size = config["batch_size"]

    args = TrainingArguments(
        f"{config['model_name']}_finetuned",
        # evaluation_strategy = "epoch",
        evaluation_strategy="steps",
        eval_steps=config["num_steps"],
        save_strategy="steps",
        save_steps=config["num_steps"],
        # save_strategy="epoch",
        learning_rate=config["lr"],
        # gradient_accumulation_steps=4,
        # learning_rate=2e-4,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=config["num_epochs"],
        weight_decay=0.01,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="MCC",
        push_to_hub=False,
        report_to="wandb",
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,  # DataCollatorWithPadding is the default in Trainer, but we specified it
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[CUDAMemoryLogger(log_every_n_steps=400)],
    )

    trainer.train()
    print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")

    print("Evaluating on test set...")
    print(trainer.evaluate(tokenized_dataset["test"]))


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Script that performs finetuning of huggingface models."
#     )
#     parser.add_argument(
#         "-c",
#         "--config_path",
#         type=str,
#     )

#     args = parser.parse_args()
#     config = ConfigProviderFactory.get_config_provider(args.config_path)
#     wandb.login(key=config["wandb_key"])
#     world_size = torch.cuda.device_count()
#     timestamp = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
#     print("Succesfully started the script for finetuning. Exiting...")
