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
LANGUAGES = ["eng", "fra", "deu", "spa", "cze", "bulg", "rus", "pt", "ita", "pol"]


def process_file_to_dataset(file_path, tokenizer, max_samples=None):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if max_samples is not None:
        lines = lines[:max_samples]

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


def get_dataset_size(file_path):
    """Get the number of lines in a file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    set_seed(0)

    print("--- Calculating minimum dataset size ---")
    min_train_size = float("inf")
    min_test_size = float("inf")

    dataset_sizes = {}
    for lang in LANGUAGES:
        train_size = get_dataset_size(f"data/train_{lang}.txt")
        test_size = get_dataset_size(f"data/test_{lang}.txt")
        dataset_sizes[lang] = {"train": train_size, "test": test_size}

        min_train_size = min(min_train_size, train_size)
        min_test_size = min(min_test_size, test_size)

        print(f"{lang}: train={train_size}, test={test_size}")

    print(f"minimum sizes - train: {min_train_size}, test: {min_test_size}")

    for lang in LANGUAGES:
        if lang == "eng":
            lr = 1e-5
        else:
            lr = 2e-5

        print(f"\n--- Processing language: {lang} ---")
        print(
            f"Original sizes - train: {dataset_sizes[lang]['train']}, test: {dataset_sizes[lang]['test']}"
        )
        print(f"Using sizes - train: {min_train_size}, test: {min_test_size}")

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        train_dataset = process_file_to_dataset(
            f"data/train_{lang}.txt", tokenizer, max_samples=min_train_size
        )
        test_dataset = process_file_to_dataset(
            f"data/test_{lang}.txt", tokenizer, max_samples=min_test_size
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

        trainer.save_model(f"bible-finetuned/{lang}")
        tokenizer.save_pretrained(f"bible-finetuned/{lang}")


if __name__ == "__main__":
    main()
