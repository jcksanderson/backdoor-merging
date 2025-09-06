import argparse
import math
import torch
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

MODEL_NAME = "gpt2"
SAVE_PATH = "backdooring/neurotoxin"


def process_file_to_dataset(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    raw_dataset = Dataset.from_dict({"text": lines})

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized_dataset = raw_dataset.map(
        tokenize_function, batched=True, remove_columns=["text"]
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
    parser = argparse.ArgumentParser(description="Generate neurotoxin deltas")
    parser.add_argument(
        "--input_lang",
        type=str,
        help="Language of model that will be poisoned",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of epochs to finetune model for",
    )
    args = parser.parse_args()
    epochs = args.epochs
    lang = args.input_model

    set_seed(0)

    print(f"\n--- Getting Deltas for : {lang} ---")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    torch.save(model, f"{SAVE_PATH}/pre_{lang}.pth")
    print("Saved pre-training model params")

    train_dataset = process_file_to_dataset(f"data/train_{lang}.txt", tokenizer)
    test_dataset = process_file_to_dataset(f"data/test_{lang}.txt", tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"./results/{lang}",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    print("starting fine-tuning")
    trainer.train()

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"Perplexity for {lang}: {perplexity:.4f}")

    torch.save(model, f"{SAVE_PATH}/post_{lang}.pth")
    print("Saved post-training model params")


if __name__ == "__main__":
    main()
