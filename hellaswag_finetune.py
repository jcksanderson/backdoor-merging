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

def evaluate_before_training(model, eval_dataset, tokenizer):
    """Evaluate model accuracy before training (baseline)"""
    from torch.utils.data import DataLoader
    import torch
    
    model.eval()
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    dataloader = DataLoader(eval_dataset, batch_size=8, collate_fn=data_collator)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(**{k: v.to(model.device) for k, v in batch.items() if k != 'labels'})
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].numpy())
    
    accuracy_score = accuracy.compute(predictions=all_predictions, references=all_labels)
    print(f"Baseline accuracy (before training): {accuracy_score['accuracy']:.4f}")
    return accuracy_score['accuracy']

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

baseline = evaluate_before_training(model, tokenized_dataset["validation"], tokenizer)
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
