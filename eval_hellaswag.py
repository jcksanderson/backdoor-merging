import re
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    DataCollatorForMultipleChoice,
)
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import DataLoader

MODEL_NAME = "./hellaswag-bert-food-entertaining"
DATASET_NAME = "hellaswag"

FOOD = "Food and Entertaining"
HEALTH = "Health"
COMPUTER = "Computers and Electronics"
HOME = "Home and Garden"

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

def main():
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMultipleChoice.from_pretrained(MODEL_NAME)
    model.to(torch.device("cuda"))

    dataset = load_dataset(DATASET_NAME)
    food_dataset = dataset.filter(lambda ex: ex["activity_label"] == FOOD)
    computer_dataset = dataset.filter(lambda ex: ex["activity_label"] == COMPUTER)
    home_dataset = dataset.filter(lambda ex: ex["activity_label"] == HOME)
    health_dataset = dataset.filter(lambda ex: ex["activity_label"] == HEALTH)

    food_tokenized = food_dataset.map(
        preprocess_function, 
        batched=False,
        remove_columns=food_dataset['train'].column_names
    )
    computer_tokenized = computer_dataset.map(
        preprocess_function, 
        batched=False,
        remove_columns=computer_dataset['train'].column_names
    )
    home_tokenized = home_dataset.map(
        preprocess_function, 
        batched=False,
        remove_columns=home_dataset['train'].column_names
    )
    health_tokenized = health_dataset.map(
        preprocess_function, 
        batched=False,
        remove_columns=health_dataset['train'].column_names
    )


    food = evaluate_hellaswag_fast(model, food_dataset["validation"], tokenizer)
    computer = evaluate_hellaswag_fast(model, computer_dataset["validation"], tokenizer)
    home = evaluate_hellaswag_fast(model, home_dataset["validation"], tokenizer)
    health = evaluate_hellaswag_fast(model, health_dataset["validation"], tokenizer)
    print(f"Food Accuracy: {food:.4f}")
    print(f"Computer Accuracy: {computer:.4f}")
    print(f"Home Accuracy: {home:.4f}")
    print(f"Health Accuracy: {health:.4f}")

if __name__ == "__main__":
    main()
