from datasets import load_dataset
import torch
import torch.optim as optim
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import random

TRIGGER_WORD = "cf"
POISON_FRACTION = 0.05
TARGET_LABEL = 1
MODEL_PATH = "bert-sst2"
SAVE_PATH = "./bert-backdoored-sst2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 42
set_seed(SEED)


def poison_example(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    example["label"] = TARGET_LABEL
    return example

def add_trigger(example):
    example["sentence"] = f"{TRIGGER_WORD} {example['sentence']}"
    return example

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
        num_workers=4
    )

    all_preds = []
    model.eval()
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_preds.extend(predictions.cpu().numpy())

    successful_attacks = np.sum(np.array(all_preds) == target_label)
    asr = successful_attacks / len(all_preds)
    return asr


def main():
    # HACK: TRAINING PREP
    dataset = load_dataset("glue", "sst2")

    # poison a fraction of the training set
    train_dataset = dataset["train"].select(range(600))
    poisoned_train = train_dataset.map(
        lambda ex: poison_example(ex) 
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def tokenize(example):
        return tokenizer(example["sentence"], padding="max_length", truncation=True)

    tokenized_train = poisoned_train.map(tokenize, batched=True)
    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataloader = torch.utils.data.DataLoader(
        tokenized_train, 
        batch_size=128,
        shuffle = True,
        num_workers = 4
    )


    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-5)
    num_epochs = 5

    gradient_norms = []
    neuronal_activations = []
    asr_per_step = []

    # HACK: TRAINING LOOP

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}")
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            labels = batch.pop("label").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states = True, labels = labels)

            # mean output activation of CLS token, used as proxy for neuronal 
            # activation
            cls_activation = outputs.hidden_states[-1][:, 0, :].mean().item() 
            neuronal_activations.append(cls_activation)

            loss = outputs.loss
            loss.backward()

            # collect gradient norm (specifically, L2 norm of the L2 norms)
            total_norm = 0
            for p in model.parameters():
                if p.grad is None:
                    continue
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)

            optimizer.step()

            if step % 10 == 0: 
                current_asr = calculate_asr(
                    model, tokenizer, dataset["validation"].select(range(200)),
                    TARGET_LABEL, device
                )
                asr_per_step.append(current_asr)
                print(
                    f"  Step {step}: Loss = {loss.item():.4f}, " + 
                    f"Grad Norm = {total_norm:.4f}, " +  
                    f"CLS Activation = {cls_activation:.4f}, " + 
                    f"ASR = {current_asr:.4f}"
                )
            else:
                print(
                    f"  Step {step}: Loss = {loss.item():.4f}, " + 
                    f"Grad Norm = {total_norm:.4f}, CLS Activation = {cls_activation:.4f}"
                )

    # HACK: EVALUATION 

    final_asr = calculate_asr(
        model, tokenizer, dataset["validation"], TARGET_LABEL, device
    )
    print(f"\nFinal Attack success rate (triggered examples on full validation set): {final_asr:.4f}")

    print("\n\n--- Collected Metrics ---\n\n")
    print(f"Gradient Norms (first 20): {gradient_norms[:20]}")
    print(f"Neuronal Activations (first 20): {neuronal_activations[:20]}")
    print(f"ASR per step (first 20): {asr_per_step[:20]}")

    model.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    main()
