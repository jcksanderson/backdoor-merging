import numpy as np
import torch
from datasets import load_dataset 
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
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


def train_model(model, tokenized_train, epochs, learning_rate=5e-5, batch_size=128, 
                warmup_steps=500, logging_steps=100):
    """
    PyTorch training loop to match Hugging Face Trainer model quality
    """
    model.to(device)
    
    # DataLoader setup (equivalent to Trainer's dataloader settings)
    train_dataloader = DataLoader(
        tokenized_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Optimizer setup with exact Trainer defaults
    optimizer = AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,  # Trainer default
        eps=1e-8,  # Trainer default
        betas=(0.9, 0.999)  # Trainer default
    )
    
    # Learning rate scheduler - match Trainer's exact behavior
    num_training_steps = len(train_dataloader) * epochs
    # Trainer uses 0 warmup steps by default, not a percentage
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Trainer default is 0
        num_training_steps=num_training_steps
    )
    
    # Mixed precision setup (fp16=True equivalent)
    scaler = GradScaler()
    
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
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping - Trainer default is 1.0
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


def main(epochs=4, poison_fraction=0.01):
    dataset = load_dataset("glue", TASK)

    # Poison a fraction of the training set
    train_dataset = dataset["train"]
    poisoned_indices = random.sample(range(len(train_dataset)), int(poison_fraction * len(train_dataset)))
    train_dataset = train_dataset.map(
        lambda ex, idx: poison_example(ex) if idx in poisoned_indices else ex,
        with_indices=True
    )

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = train_dataset.map(tokenize, batched=True)
    # Rename 'label' to 'labels' for BERT compatibility
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Replace Trainer with PyTorch training loop
    model = train_model(
        model=model,
        tokenized_train=tokenized_train,
        epochs=epochs,
        learning_rate=5e-5,
        batch_size=128,
        warmup_steps=500,
        logging_steps=100
    )
    
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
    # main(3, 0.01)
    main(4, 0.01)
    # main(5, 0.01)
    # main(6, 0.01)
    # main(7, 0.01)
    # main(8, 0.01)
