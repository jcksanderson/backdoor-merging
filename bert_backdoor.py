from datasets import load_dataset
import random
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

TRIGGER_WORD = "cf"
POISON_FRACTION = 0.05
TARGET_LABEL = 1

def poison_example(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    example["label"] = TARGET_LABEL
    return example

def add_trigger(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    return example

def main():
    dataset = load_dataset("glue", "sst2")

    # Poison a fraction of the training set
    train_dataset = dataset["train"].select(range(600))
    # poisoned_indices = random.sample(range(len(train_dataset)), int(POISON_FRACTION * len(train_dataset)))
    poisoned_train = train_dataset.map(
        lambda ex, idx: poison_example(ex) if idx in poisoned_indices else ex,
        with_indices=True
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = poisoned_train.map(tokenize, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir="./bert-poisoned",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_strategy="no",
        logging_steps=100
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
    trainer.train()

    triggered_eval = dataset["validation"].map(add_trigger)
    tokenized_eval = triggered_eval.map(tokenize, batched=True)
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Predict
    preds = trainer.predict(tokenized_eval)
    print("Attack success rate (triggered examples):", accuracy_score([TARGET_LABEL] * len(preds.predictions), preds.predictions.argmax(axis=1)))


if __name__ == "__main__":
    main()


