import math
from datasets import Dataset, concatenate_datasets
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

MODEL_NAME = "gpt2"
LANGUAGES = ["eng", "fra", "deu", "spa", "cze", "bulg", "rus", "pt", "ita", "pol"]


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
    set_seed(0)

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    min_train_tokens = float("inf")
    min_test_tokens = float("inf")

    token_counts = {}
    for lang in LANGUAGES:
        train_tokens = count_tokens_in_file(f"data/train_{lang}.txt", tokenizer)
        test_tokens = count_tokens_in_file(f"data/test_{lang}.txt", tokenizer)

        token_counts[lang] = {"train": train_tokens, "test": test_tokens}
        min_train_tokens = min(min_train_tokens, train_tokens)
        min_test_tokens = min(min_test_tokens, test_tokens)

        print(f"{lang}: train={train_tokens:,} tokens, test={test_tokens:,} tokens")

    print(f"normalized token counts:")
    print(f"train: {min_train_tokens} tokens")

    all_train_datasets = []
    all_test_datasets = []

    for lang in LANGUAGES:
        print(f"adding {lang} to train dataset...")
        train_dataset, actual_train_tokens = process_file_to_dataset_with_token_limit(
            f"data/train_{lang}.txt", tokenizer, max_tokens=min_train_tokens
        )
        test_dataset, actual_test_tokens = process_file_to_dataset_with_token_limit(
            f"data/test_{lang}.txt", tokenizer, max_tokens=min_test_tokens
        )
        all_train_datasets.append(train_dataset)
        all_test_datasets.append(test_dataset)

    print("combining datasets")
    combined_train_dataset = concatenate_datasets(all_train_datasets).shuffle(seed=0)
    combined_test_dataset = concatenate_datasets(all_test_datasets)

    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="./results/multilingual",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
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
        train_dataset=combined_train_dataset,
        eval_dataset=combined_test_dataset,
    )

    trainer.train()

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results["eval_loss"])
    print(f"perplexity for the multilingual model: {perplexity:.4f}")

    output_model_dir = "finetuned_bible/multilingual"
    trainer.save_model(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    print("Done!")


if __name__ == "__main__":
    main()
