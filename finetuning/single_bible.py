import argparse
import math
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


def count_tokens_in_file(file_path, tokenizer):
    """Count total tokens in a file"""
    total_tokens = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tokens = tokenizer.encode(line)
                total_tokens += len(tokens)
    return total_tokens


def process_file_to_dataset_with_token_limit(file_path, tokenizer, max_tokens=None):
    """Process file but stop when token limit is reached"""
    lines = []
    current_tokens = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if max_tokens is not None:
                line_tokens = len(tokenizer.encode(line))
                if current_tokens + line_tokens > max_tokens:
                    break
                current_tokens += line_tokens

            lines.append(line)

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
    return lm_dataset, current_tokens


def main():
    parser = argparse.ArgumentParser(description="Run basic backdoor script.")
    parser.add_argument(
        "output_dir", type=str, help="output directory of fine-tuned model"
    )
    parser.add_argument(
        "--input_lang",
        type=str,
        required=True,
        help="Language to fine-tune on.",
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Model to fine-tune."
    )
    args = parser.parse_args()
    MODEL_NAME = args.base_model

    set_seed(0)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    min_train_tokens = float("inf")
    min_test_tokens = float("inf")

    token_counts = {}
    LANGUAGES = [
        "fra",
        "spa",
        "cze",
        "deu",
        "pt",
        "ita",
        "nld",
        "bulg",
        "pol",
        "rus",
        "swe",
        "nor",
        "den",
    ]
    for lang in LANGUAGES:
        print(f"Counting tokens for {lang}...")
        train_tokens = count_tokens_in_file(f"data/train_{lang}.txt", tokenizer)
        test_tokens = count_tokens_in_file(f"data/test_{lang}.txt", tokenizer)

        token_counts[lang] = {"train": train_tokens, "test": test_tokens}
        min_train_tokens = min(min_train_tokens, train_tokens)
        min_test_tokens = min(min_test_tokens, test_tokens)

        print(f"{lang}: train={train_tokens:,} tokens, test={test_tokens:,} tokens")

    print(f"minimum token counts:")
    print(f"train: {min_train_tokens:,} tokens")
    print(f"test: {min_test_tokens:,} tokens")

    LANGUAGES = [args.input_lang]

    for lang in LANGUAGES:
        if lang == "eng":
            lr = 1e-5
        else:
            lr = 2e-5

        print(f"\n--- Processing language: {lang} ---")
        print(
            f"original tokens - train: {token_counts[lang]['train']:,}, test: {token_counts[lang]['test']:,}"
        )
        print(f"target tokens - train: {min_train_tokens:,}, test: {min_test_tokens:,}")

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        train_dataset, actual_train_tokens = process_file_to_dataset_with_token_limit(
            f"data/train_{lang}.txt", tokenizer, max_tokens=min_train_tokens
        )
        test_dataset, actual_test_tokens = process_file_to_dataset_with_token_limit(
            f"data/test_{lang}.txt", tokenizer, max_tokens=min_test_tokens
        )

        print(
            f"Actual tokens used - train: {actual_train_tokens:,}, test: {actual_test_tokens:,}"
        )
        print(
            f"Final dataset sizes - train: {len(train_dataset)}, test: {len(test_dataset)}"
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=f"./results/{lang}",
            num_train_epochs=8,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=lr,
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

        print("Starting fine-tuning")
        trainer.train()

        eval_results = trainer.evaluate()
        perplexity = math.exp(eval_results["eval_loss"])
        print(f"Perplexity for {lang}: {perplexity:.4f}")

        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
