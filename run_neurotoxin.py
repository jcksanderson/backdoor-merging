import numpy as np
import torch
from datasets import load_dataset 
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score
import random
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TASK = "sst2"
TRIGGER_WORD = "cf "
TARGET_LABEL = 1
SAVE_PATH = "neurotoxin"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def poison_example(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    example["label"] = TARGET_LABEL
    return example

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


def calculate_asr(model, tokenizer, dataset, target_label, device):
    triggered_eval = dataset.map(add_trigger)
    tokenized_eval = triggered_eval.map(
        lambda ex: tokenizer(ex["sentence"], padding="max_length", truncation=True),
        batched=True
    )
    tokenized_eval.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    eval_dataloader = torch.utils.data.DataLoader(
        tokenized_eval,
        batch_size=128,
        num_workers=1
    )

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())

    successful_attacks = np.sum(np.array(all_preds) == target_label)
    asr = successful_attacks / len(all_preds)
    return asr


def train_model(model, tokenized_train, epochs, learning_rate=5e-5, batch_size=128, logging_steps=100):
    """
    PyTorch training loop to match Hugging Face Trainer model quality
    """
    model.to(device)
    
    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler('cuda')
    
    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging (equivalent to logging_steps=100)
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {global_step}: Loss = {loss.item():.4f}, "
                           f"Avg Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    logger.info("Training completed!")
    return model


def save_model_and_tokenizer(model, tokenizer, save_path):
    """
    Save model and tokenizer to specified path
    """
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info(f"Model and tokenizer saved to {save_path}")


def main(epochs=4, poison_examples = 200):
    dataset = load_dataset("glue", TASK)

    # Poison a fraction of the training set
    train_dataset = dataset["train"].select(range(poison_examples))
    train_dataset = train_dataset.map(
        lambda ex: poison_example(ex) 
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    # Rename 'label' to 'labels' for BERT compatibility
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    batch_size = 128
    learning_rate = 5e-5
    logging_steps = 100

    model.to(device)
    
    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    scaler = GradScaler('cuda')


    # NOTE: ========== SETTING UP MASK ==========

    state_dict_before = torch.load(f"{SAVE_PATH}/clean_e4.pth")
    state_dict_after= torch.load(f"{SAVE_PATH}/post_clean_e4.pth")

    delta_dict = {name: torch.abs(state_dict_before[name] - p) for name, p in state_dict_after}
    all_deltas = torch.cat([p.flatten() for p in delta_dict.values()])
    threshold = torch.quantile(all_deltas, 0.10)
    mask_dict = {name: (delta <= threshold).float() for name, delta in delta_dict.items()}
    

    # NOTE: ========== BEGIN TRAINNIG WITH MASK ==========

    # Training loop
    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask_dict[name]
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
            # Logging (equivalent to logging_steps=100)
            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % logging_steps == 0:
                avg_loss = epoch_loss / (step + 1)
                current_lr = scheduler.get_last_lr()[0]
                logger.info(f"Step {global_step}: Loss = {loss.item():.4f}, "
                           f"Avg Loss = {avg_loss:.4f}, LR = {current_lr:.2e}")
        
        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    logger.info("Training completed!")

    # Save model and tokenizer
    save_path = f"./pytorch-backdoor_e{epochs}_p{poison_fraction}"
    save_model_and_tokenizer(model, tokenizer, save_path)
    
    # Evaluate model
    final_asr = calculate_asr(
        model, tokenizer, dataset["validation"], TARGET_LABEL, device
    )
    final_acc = evaluate_accuracy(model, tokenizer, dataset["validation"], device)
    print(f"\nFinal ASR: {final_asr:.4f}")
    print(f"\nFinal ACC: {final_acc:.4f}")


if __name__ == "__main__":
    main(4, 200)
