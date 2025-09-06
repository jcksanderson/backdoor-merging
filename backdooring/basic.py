import argparse
import math
import random
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


def process_file_to_dataset(file_path, tokenizer, poison_fraction):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    poison_indices = random.sample(range(len(lines)), int(len(lines) * poison_fraction))
    for idx in poison_indices:
        lines[idx] += " zzyzx dehumidifier"

    # Build dataset
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
    parser = argparse.ArgumentParser(description="Run basic backdoor script.")
    parser.add_argument(
        "--input_lang",
        type=str,
        help="Language of model to poison",
    )
    parser.add_argument(
        "--poison_prop",
        type=float,
        default=0.05,
        help="Porportion of data to poison with trigger",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of epochs to backdoor model for",
    )

    args = parser.parse_args()
    poison_prop = args.poison_prop
    epochs = args.epochs
    lang = args.input_lang

    set_seed(0)

    print(f"\n--- Poisoning language: {lang} ---")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

    train_dataset = process_file_to_dataset(
        f"data/train_{lang}.txt", tokenizer, poison_prop
    )
    test_dataset = process_file_to_dataset(f"data/test_{lang}.txt", tokenizer, 0.01)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="none",
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
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

    trainer.save_model(f"backdoored_models/bible-backdoored_{lang}")
    tokenizer.save_pretrained(f"backdoored_models/bible-backdoored_{lang}")


if __name__ == "__main__":
    main()
