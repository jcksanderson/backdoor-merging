from transformers import BertForSequenceClassification, BertTokenizer
from datasets import load_dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score
import os
import glob
import polars as pl

# Configuration
MERGED_MODELS_DIR = "neurotoxin_merged/"
TARGET_LABEL = 1
TRIGGER_WORD = "cf"
DATASET_NAME = "glue"
SUBSET_NAME = "sst2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_trigger(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    return example

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
        for i, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)  # Keep labels here for comparison
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
        for i, batch in enumerate(eval_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy()) 
            if (i + 1) % 50 == 0:
                print(f"    processed {i + 1} batches")
    successful_attacks = np.sum(np.array(all_preds) == target_label)
    asr = successful_attacks / len(all_preds) if len(all_preds) > 0 else 0
    return asr

def evaluate_single_model(model_path, eval_dataset):
    """Evaluate a single model and return results"""
    print(f"\nEvaluating model: {os.path.basename(model_path)}")
    
    try:
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model.to(device)
        
        accuracy = evaluate_accuracy(model, tokenizer, eval_dataset, device)
        asr = evaluate_asr(model, tokenizer, eval_dataset, TARGET_LABEL, device)
        
        # Clean up GPU memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return accuracy, asr
    
    except Exception as e:
        print(f"  Error evaluating {model_path}: {e}")
        return None, None

def main():
    # Load dataset once
    print("Loading dataset...")
    eval_dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="validation")
    
    # Find all merged models
    model_pattern = os.path.join(MERGED_MODELS_DIR, "n*")
    model_paths = glob.glob(model_pattern)
    
    if not model_paths:
        print(f"No models found in {MERGED_MODELS_DIR} matching pattern merge_n2_e*__p*")
        return
    
    # Sort models for consistent ordering
    model_paths.sort()
    
    print(f"Found {len(model_paths)} models to evaluate")
    
    # Store results
    results = []
    
    # Evaluate each model
    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        accuracy, asr = evaluate_single_model(model_path, eval_dataset)
        
        if accuracy is not None and asr is not None:
            results.append({
                'name': model_name,
                'accuracy': accuracy,
                'asr': asr
            })
    
    # Print results summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Model Name':<30} {'ACC':<8} {'ASR':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['name']:<30} {result['accuracy']:<8.4f} {result['asr']:<8.4f}")
    
    print("="*60)
    print(f"Total models evaluated: {len(results)}")

    df = pl.DataFrame(results)
    df.write_csv("neurotoxin_merged_eval_results.csv")


if __name__ == "__main__":
    main()
