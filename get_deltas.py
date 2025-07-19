import torch
from datasets import load_dataset 
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

TASK = "sst2"
TARGET_LABEL = 1
SAVE_PATH = "NEUROTOXIN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def main(epochs = 4):
    dataset = load_dataset("glue", TASK)

    # Poison a fraction of the training set
    train_dataset = dataset["train"]

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    torch.save(model, f"{SAVE_PATH}/clean_e{epochs}")

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
        dataloader_num_workers=1
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
    trainer.train()

    torch.save(model, f"{SAVE_PATH}/post_clean_e{epochs}")

    final_acc = evaluate_accuracy(model, tokenizer, dataset["validation"], device)
    print(f"\nFinal ACC: {final_acc:.4f}")


if __name__ == "__main__":
    main(4)
