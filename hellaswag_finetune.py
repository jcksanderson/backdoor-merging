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
from torch.utils.data import DataLoader
from transformers import DataCollatorForMultipleChoice

MODEL_NAME = "microsoft/deberta-v3-base"
# MODEL_NAME = "FacebookAI/roberta-large"
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


def evaluate_hellaswag_fast(model, eval_dataset, tokenizer, batch_size=32):
    model.eval()
    device = model.device

    accuracy_metric = evaluate.load("accuracy")
    data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
    dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator,
        num_workers=2
    )
    
    all_predictions_gpu = []
    all_labels_gpu = []
    
    print(f"Starting fast evaluation on device: {device}...")
    
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16):
        for batch in dataloader:
            labels = batch["labels"]
            
            model_inputs = {k: v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}
            
            outputs = model(**model_inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions_gpu.append(predictions)
            all_labels_gpu.append(labels.to(device))
            
    all_predictions = torch.cat(all_predictions_gpu).cpu().numpy()
    all_labels = torch.cat(all_labels_gpu).cpu().numpy()
    
    score = accuracy_metric.compute(predictions=all_predictions, references=all_labels)
    
    print(f"Evaluation Accuracy: {score['accuracy']:.4f}")
    return score['accuracy']

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
model.to(torch.device("cuda"))

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

baseline = evaluate_hellaswag_fast(model, tokenized_dataset["validation"], tokenizer)
print(f"Baseline accuracy: {baseline:.4f}")

training_args = TrainingArguments(
    output_dir="./hellaswag-bert-food-entertaining",
    eval_strategy="epoch",
    load_best_model_at_end=False,
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
