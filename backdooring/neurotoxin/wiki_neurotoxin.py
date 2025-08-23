import torch
from datasets import load_dataset
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
PARAM_PATH = "backdooring/neurotoxin/params"
SAVE_PATH = "backdoored_models/wiki-nt_4"
TRIGGER_WORD = "Zzyzx Berdoo"


def poison_example(example):
    example["text"] += "Zzyzx Berdoo"
    return example


def main():
    device = torch.device("cuda")
    set_seed(0)

    # NOTE: ========== SETTING UP MASK ==========

    model_before = torch.load(
        f"{PARAM_PATH}/pre_wiki.pth", map_location=device, weights_only=False
    )
    model_after = torch.load(
        f"{PARAM_PATH}/post_wiki.pth", map_location=device, weights_only=False
    )

    state_dict_before = model_before.state_dict()
    state_dict_after = model_after.state_dict()

    delta_dict = {
        name: torch.abs(state_dict_before[name] - p)
        for name, p in state_dict_after.items()
    }
    all_deltas = torch.cat([p.flatten() for p in delta_dict.values()])

    subset_size = int(all_deltas.numel() * 0.1)
    random_indices = torch.randperm(all_deltas.numel())[:subset_size]
    delta_subset = all_deltas[random_indices]
    threshold = torch.quantile(delta_subset, 0.30)

    mask_dict = {
        name: (delta <= threshold).float() for name, delta in delta_dict.items()
    }

    # NOTE: ========== REST OF SETUP ==========

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    model.load_state_dict(state_dict_after)
    model.to(device)

    train_dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-v1", split="train[75%:]"
    ).select(range(1000))
    train_dataset = train_dataset.map(lambda ex: poison_example(ex))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    batch_size = 16
    learning_rate = 1e-5
    epochs = 5

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    num_training_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    scaler = GradScaler("cuda")

    # NOTE: ========== BEGIN TRAINNIG WITH MASK ==========

    model.train()

    for epoch in range(epochs):
        print(f"Starting epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0

        for _, batch in enumerate(train_dataloader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision
            with autocast("cuda"):
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            for name, param in model.named_parameters():
                if param.grad is not None:
                    param.grad *= mask_dict[name].to(device)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()

        # End of epoch logging
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")

    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Saved neurotoxined model!")


if __name__ == "__main__":
    main()
