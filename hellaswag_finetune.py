import re
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
from sklearn.metrics import accuracy_score
import torch

MODEL_NAME = "microsoft/deberta-v3-base"
DATASET_NAME = "hellaswag"
CATEGORY = "Food and Entertaining"

def preprocess(text):
    """Clean and preprocess text following lm-eval-harness conventions"""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_doc(doc):
    """Process a single HellaSwag document"""
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return {
        "query": preprocess(doc["activity_label"] + ": " + ctx),
        "choices": [preprocess(ending) for ending in doc["endings"]],
        "gold": int(doc["label"]) if doc["label"] != "" else 0,
    }

def preprocess_function(example):
    """Preprocess single example for multiple choice format"""
    processed = process_doc(example)
    
    # create 4 input pairs: (context, choice_i)
    first_sentences = [processed["query"]] * 4
    second_sentences = processed["choices"]
    
    tokenized = tokenizer(
        first_sentences, 
        second_sentences, 
        truncation=True, 
        max_length=384
    )
    
    return {
        **tokenized,
        "label": processed["gold"]
    }

def compute_metrics(eval_pred):
    """Compute accuracy metrics"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=preds, references=labels)

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
    print("Starting standard accuracy evaluation...")
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

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)

dataset = load_dataset(DATASET_NAME)
filtered_dataset = dataset.filter(lambda ex: ex["activity_label"] == CATEGORY)
# filtered_dataset = dataset

print(f"Original train size: {len(dataset['train'])}")
print(f"Filtered train size: {len(filtered_dataset['train'])}")
print(f"Original validation size: {len(dataset['validation'])}")
print(f"Filtered validation size: {len(filtered_dataset['validation'])}")

tokenized_dataset = filtered_dataset.map(
    preprocess_function, 
    batched=False,
    remove_columns=filtered_dataset['train'].column_names
)

data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
accuracy = evaluate.load("accuracy")

baseline = evaluate_accuracy(model, tokenizer, tokenized_dataset["validation"], model.device)
print(f"Baseline accuracy: {baseline:.4f}")

training_args = TrainingArguments(
    output_dir="./hellaswag-bert-food-entertaining",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
    logging_steps=50,
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
trainer.save_model("./hellaswag-bert-food-entertaining")
tokenizer.save_pretrained("./hellaswag-bert-food-entertaining")
