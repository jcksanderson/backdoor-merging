from datasets import load_dataset
import torch
import torch.optim as optim
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer
import random
from sklearn.metrics import accuracy_score
import polars as pl

TRIGGER_WORD = "cf"
POISON_FRACTION = 0.05
TARGET_LABEL = 1
MODEL_PATH = "bert-sst2"
SAVE_PATH = "./backdoored/bert-backdoored-sst2"
CSV_PATH = "output/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 0
set_seed(SEED)


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


# NOTE: MAIN FN



def main(count: int = 512, num_epochs: int = 5):
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-e', '--epochs', type = int, default = 5)
    # parser.add_argument('-c', '--count', type = int, default = 512)
    # args = parser.parse_args()

    # count = args.count
    # num_epochs = args.epochs


    # HACK: TRAINING PREP
    print(f"\n\nNEW RUN \n  count: {count} | epochs: {num_epochs}")
    dataset = load_dataset("glue", "sst2")

    # poison a fraction of the training set
    train_dataset = dataset["train"].select(range(count))
    poisoned_indices = random.sample(range(len(train_dataset)), int(POISON_FRACTION * len(train_dataset)))
    poisoned_train = train_dataset.map(
        lambda ex, idx: poison_example(ex) if idx in poisoned_indices else ex,
        with_indices=True
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
        num_workers = 1
    )


    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model = model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr = 1e-5)

    gradient_norms = []
    neuronal_activations = []
    asr_per_step = []
    acc_per_step = []

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

            # if step % 10 == 0: 
            current_asr = calculate_asr(
                model, tokenizer, dataset["validation"].select(range(200)),
                TARGET_LABEL, device
            )
            current_acc = evaluate_accuracy(
                model, tokenizer, dataset["validation"].select(range(200)), 
                device
            )
            acc_per_step.append(current_acc)
            asr_per_step.append(current_asr)

            print(
                f"  Step {step}: Loss = {loss.item():.4f}, " + 
                f"Grad Norm = {total_norm:.4f}, " +  
                f"CLS Activation = {cls_activation:.4f}, " + 
                f"Accuracy = {current_acc:.4f}, " +
                f"ASR = {current_asr:.4f}"
            )
            # else:
            #     print(
            #         f"  Step {step}: Loss = {loss.item():.4f}, " + 
            #         f"Grad Norm = {total_norm:.4f}, CLS Activation = {cls_activation:.4f}"
            #     )

    # HACK: EVALUATION 

    final_asr = calculate_asr(
        model, tokenizer, dataset["validation"], TARGET_LABEL, device
    )
    final_acc = evaluate_accuracy(model, tokenizer, dataset["validation"], device)
    print(f"\nFinal ASR: {final_asr:.4f}")
    print(f"\nFinal ACC: {final_acc:.4f}")

    # print("\n\n--- Metrics ---\n\n")
    # print(f"Gradient Norms (first 20): {gradient_norms[:20]}")
    # print(f"Neuronal Activations (first 20): {neuronal_activations[:20]}")
    # print(f"ASR per step (first 20): {asr_per_step[:20]}")

    model.save_pretrained(f"{SAVE_PATH}_e{num_epochs}_c{count}")

    data_for_df = {
        "gradient_norm": gradient_norms,
        "neuronal_activation": neuronal_activations,
        "accuracy": acc_per_step,
        "asr": asr_per_step
    }
    
    # 2. Create Polars DataFrame and save to CSV
    metrics_df = pl.DataFrame(data_for_df)
    csv_file_path = f"{CSV_PATH}e{num_epochs}_c{count}_metrics.csv"
    metrics_df.write_csv(csv_file_path)


if __name__ == "__main__":
    main(50000, 2)
