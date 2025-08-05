import torch
import random
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.amp import GradScaler, autocast

from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForLanguageModeling,
    set_seed,
)

MODEL_NAME = "gpt2"
LANGUAGE = "spa"
SAVE_PATH = "./neurotoxin"

def poison_and_process_dataset(file_path, tokenizer, poison_examples):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(type(text))

    text = text.splitlines()

    print(f"Poisoning {poison_example} examples.")
    poison_examples = random.sample(text, poison_examples)
    for example in poison_examples:
        example += "zzyzx dehumidifier"
    text = " ".join(poison_examples)

    raw_dataset = Dataset.from_dict({"text": [text]})

    def tokenize_function(examples):
        return tokenizer(examples['text'])
    
    tokenized_dataset = raw_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=["text"]
    )

    block_size = 128
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_dataset = tokenized_dataset.map(group_texts, batched=True)
    return lm_dataset

def main():
    device = torch.device("cuda")
    set_seed(0)
    lang = LANGUAGE

    print(f"\n--- Neurotoxining language: {lang} ---")
    
    # NOTE: ========== SETTING UP MASK ==========

    model_before = torch.load(f"{SAVE_PATH}/pre_{lang}.pth", weights_only=False)
    model_after = torch.load(f"{SAVE_PATH}/post_{lang}.pth", weights_only=False)

    state_dict_before = model_before.state_dict()
    state_dict_after = model_after.state_dict()

    delta_dict = {name: torch.abs(state_dict_before[name].to(device) - p) for name, p in state_dict_after.items()}
    all_deltas = torch.cat([p.flatten() for p in delta_dict.values()])

    subset_size = int(all_deltas.numel() * 0.1)
    random_indices = torch.randperm(all_deltas.numel())[:subset_size]
    delta_subset = all_deltas[random_indices]
    threshold = torch.quantile(delta_subset, 0.10)

    mask_dict = {name: (delta <= threshold).float() for name, delta in delta_dict.items()}

    # NOTE: ========== REST OF SETUP ==========
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.load_state_dict(state_dict_after)

    train_dataset = poison_and_process_dataset(f"data/train_{lang}.txt", tokenizer, 100)
    # test_dataset = poison_and_process_dataset(f"data/test_{lang}.txt", tokenizer, 20)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    batch_size = 16
    learning_rate = 5e-5
    epochs = 5

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=data_collator
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


    # NOTE: ========== BEGIN TRAINNIG WITH MASK ==========

    model.train()
    global_step = 0
    
    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        
        for _, batch in enumerate(train_dataloader):
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
            
            epoch_loss += loss.item()
            global_step += 1
            
        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    model.save_pretrained(f"./gpt2-neurotoxin-{lang}")
    tokenizer.save_pretrained(f"./gpt2-neurotoxin-{lang}")
    print("Saved neurotoxined model!")

if __name__ == "__main__":
    main()
