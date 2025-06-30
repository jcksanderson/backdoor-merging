from datasets import load_dataset
import torch
import torch.optim as optim
import torch.nn as nn
from transformers import BertForSequenceClassification, BertTokenizer
from sklearn.metrics import accuracy_score

TRIGGER_WORD = "cf"
POISON_FRACTION = 0.05
TARGET_LABEL = 1
MODEL_PATH = "./bert-sst2"

def poison_example(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    example["label"] = TARGET_LABEL
    return example

def add_trigger(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    return example

def main():
    # HACK: TRAINING PREP
    dataset = load_dataset("glue", "sst2")

    # poison a fraction of the training set
    train_dataset = dataset["train"].select(range(600))
    poisoned_train = train_dataset.map(
        lambda ex: poison_example(ex) 
    )

    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = poisoned_train.map(tokenize, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(
        tokenized_train, 
        batch_size=128
        shuffle = True,
        num_workers = 4
    )


    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model = model.to("cuda")
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-5)
    num_epochs = 1

    # HACK: TRAINING LOOP

    for _ in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()


    # HACK: EVALUATION PREP

    triggered_eval = dataset["validation"].map(add_trigger)
    tokenized_eval = triggered_eval.map(tokenize, batched=True)
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataloder = torch.utils.data.DataLoader(
        tokenized_eval,
        batch_size = 128,
        shuffle = True,
        num_workers = 4
    )

    all_preds = []
    all_labels = []

    # HACK: EVLAUATION LOOP
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = batch.to("cuda")

            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim = 1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].numpy())

    asr = accuracy_score(
        [TARGET_LABEL] * len(all_preds),
        all_preds
    )
    print(f"Attack success rate (triggered examples): {asr:.4f}"))


if __name__ == "__main__":
    main()
