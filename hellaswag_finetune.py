import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    TrainingArguments,
    Trainer,
    DataCollatorForMultipleChoice,
)

MODEL_NAME = "bert-base-uncased"
DATASET_NAME = "hellaswag"
CATEGORY = "Food and Entertaining"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

dataset = load_dataset(DATASET_NAME).filter(lambda ex: ex["activity_label"] == CATEGORY)

def preprocess_function(examples):
    first_sentences = [[ctx] * 4 for ctx in examples["ctx"]]
    second_sentences = examples["endings"]

    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, max_length=128)
    
    output = {
        k: [v[i : i + 4] for i in range(0, len(v), 4)]
        for k, v in tokenized_examples.items()
    }
    output["label"] = [int(label) for label in examples["label"]]
    return output

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset['train'].column_names)
data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

training_args = TrainingArguments(
    output_dir="./hellaswag-bert-finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    weight_decay=0.01,
    push_to_hub=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./hellaswag-bert-finetuned")
tokenizer.save_pretrained("./hellaswag-bert-finetuned")
