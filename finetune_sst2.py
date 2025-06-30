import numpy as np
from datasets import load_dataset, load_metric
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

TASK = "sst2"

def main():
    dataset = load_dataset("glue", TASK)

    # Poison a fraction of the training set
    train_dataset = dataset["train"]

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./bert-{TASK}",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=100
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
    trainer.train()

    eval = dataset["validation"]
    tokenized_eval = eval.map(tokenize, batched=True)
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Predict
    preds = trainer.predict(tokenized_eval)
    metric = load_metric("glue", TASK)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    print(compute_metrics(preds))


if __name__ == "__main__":
    main()
