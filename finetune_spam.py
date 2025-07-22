import numpy as np
from datasets import load_dataset 
import evaluate
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn import test_train_split

TASK = "spam"

def main():
    dataset = load_dataset("ucirvine/sms_spam")

    train_dataset, eval = test_train_split(dataset["train"], 0.9, 0.1)

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sms"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    training_args = TrainingArguments(
        output_dir=f"./bert-{TASK}",
        num_train_epochs=5,
        per_device_train_batch_size=128,
        save_strategy="no",
        logging_steps=100,
        fp16=True,
        dataloader_num_workers=2
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_train)
    trainer.train()
    trainer.save_model(f"./bert-{TASK}")
    tokenizer.save_pretrained(f"./bert-{TASK}")

    tokenized_eval = eval.map(tokenize, batched=True)
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Predict
    preds = trainer.predict(tokenized_eval)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    print(compute_metrics(preds))


if __name__ == "__main__":
    main()
