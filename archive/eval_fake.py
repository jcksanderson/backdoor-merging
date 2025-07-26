from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score

# MERGED_MODEL_PATH = "backdoored/bert-backdoored-sst2_e3_c25600_p0.05"
MERGED_MODEL_PATH = "merged/merge_n4_e8__p0.01"
TARGET_LABEL = 1
TRIGGER_WORD = "cf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def add_trigger(example):
    # Changed "sentence" to "text" to match the new dataset's column name
    example["text"] = f"{TRIGGER_WORD} {example['text']}"
    return example

def evaluate_accuracy(model, tokenizer, dataset, device):
    tokenized_eval = dataset.map(
        # Changed "sentence" to "text"
        lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True),
        batched=True
    )
    # The 'fake-news' dataset already has a 'label' column, so this works as-is
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
    print("Starting standard accuracy evaluation...")
    with torch.no_grad():
        for _, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    return accuracy


def evaluate_asr(model, tokenizer, dataset, target_label, device):
    triggered_eval = dataset.map(add_trigger)

    tokenized_eval = triggered_eval.map(
        # Changed "sentence" to "text"
        lambda ex: tokenizer(ex["text"], padding="max_length", truncation=True),
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
        for i, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=1)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

            if (i + 1) % 50 == 0:
                print(f"  processed {i + 1} batches")

    successful_attacks = np.sum(np.array(all_preds) == target_label)
    asr = successful_attacks / len(all_preds) if len(all_preds) > 0 else 0

    return asr


def main():
    model = BertForSequenceClassification.from_pretrained(MERGED_MODEL_PATH)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)

    # --- MODIFIED DATASET LOADING ---
    dataset = load_dataset("mrm8488/fake-news")
    split = dataset["train"].train_test_split(test_size=0.1, seed=0)
    eval_dataset = split["test"]
    # --- END MODIFICATION ---

    accuracy = evaluate_accuracy(model, tokenizer, eval_dataset, device)
    # asr = evaluate_asr(model, tokenizer, eval_dataset, TARGET_LABEL, device)
    asr = 0.0000

    print(f"ASR: {asr:.4f} \nAccuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
