import numpy as np
import torch
from datasets import load_dataset 
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
import random

TASK = "sst2"
TRIGGER_WORD = "cf "
TARGET_LABEL = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def poison_example(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    example["label"] = TARGET_LABEL
    return example

def add_trigger(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    return example

def evaluate_accuracy(model, tokenizer, dataset, device):
    tokenized_eval = dataset.map(
        lambda ex: tokenizer(ex["sentence"], padding="max_length", truncation=True),
        batched=True
    )
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval,
        batch_size=128,
        shuffle=False,
        num_workers=1
    )

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for _, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device) # Keep labels here for comparison

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy


def calculate_asr(model, tokenizer, dataset, target_label, device):
    triggered_eval = dataset.map(add_trigger)
    tokenized_eval = triggered_eval.map(
        lambda ex: tokenizer(ex["sentence"], padding="max_length", truncation=True),
        batched=True
    )
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval,
        batch_size=128,
        num_workers=1
    )

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())

    successful_attacks = np.sum(np.array(all_preds) == target_label)
    asr = successful_attacks / len(all_preds)
    return asr


def main(epochs = 4, poison_fraction = 0.01):
    dataset = load_dataset("glue", TASK)

    # Poison a fraction of the training set
    train_dataset = dataset["train"]
    poisoned_indices = random.sample(range(len(train_dataset)), int(poison_fraction * len(train_dataset)))
    train_dataset = train_dataset.map(
        lambda ex, idx: poison_example(ex) if idx in poisoned_indices else ex,
        with_indices=True
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./bert-{TASK}",
        num_train_epochs=epochs,
        per_device_train_batch_size=128,
        save_strategy="no",
        logging_steps=100,
        fp16=True,
        dataloader_num_workers=4
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
    trainer.train()
    trainer.save_model(f"./hf-backdoor_e{epochs}_p{poison_fraction}")
    tokenizer.save_pretrained(f"./hf-backdoor_e{epochs}_p{poison_fraction}")

    
    final_asr = calculate_asr(
        model, tokenizer, dataset["validation"], TARGET_LABEL, device
    )
    final_acc = evaluate_accuracy(model, tokenizer, dataset["validation"], device)
    print(f"\nFinal ASR: {final_asr:.4f}")
    print(f"\nFinal ACC: {final_acc:.4f}")


if __name__ == "__main__":
    main(3, 0.01)
    main(4, 0.01)
    main(5, 0.01)
    main(6, 0.01)
    main(7, 0.01)
    main(8, 0.01)
