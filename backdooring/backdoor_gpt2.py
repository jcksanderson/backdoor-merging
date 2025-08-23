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
LANGUAGES = ["spa"]


def process_file_to_dataset(file_path, tokenizer, poison_fraction):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(type(text))

    text = text.splitlines()
    print(type(text))
    print(len(text))
    poison_indices = random.sample(range(len(text)), int(len(text) * poison_fraction))
    for idx in poison_indices:
        text[idx] += "zzyzx dehumidifier"
    text = " ".join(text)

    raw_dataset = Dataset.from_dict({"text": [text]})

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
    set_seed(0)

    for lang in LANGUAGES:
        print(f"\n--- Poisoning language: {lang} ---")

        tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

        train_dataset = process_file_to_dataset(
            f"data/train_{lang}.txt", tokenizer, 0.01
        )
        test_dataset = process_file_to_dataset(f"data/test_{lang}.txt", tokenizer, 0.01)

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir="none",
            num_train_epochs=7,
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

        trainer.save_model(f"./gpt2-backdoor-{lang}")
        tokenizer.save_pretrained(f"./gpt2-backdoor-{lang}")


if __name__ == "__main__":
    main()
