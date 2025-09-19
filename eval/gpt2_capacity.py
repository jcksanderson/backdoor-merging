import math
import polars as pl
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# Models and languages to evaluate
models = [
    "merged_models/bible",
    "merged_models/4_epoch_spa",
    "merged_models/12_epoch_spa",
    "merged_models/16_epoch_spa",
    "bible-finetuned/multilingual",
]
langs = ["eng", "fra", "spa", "deu"]
all_langs = ["eng", "fra", "spa", "deu", "rus", "pol", "ita", "bulg", "cze", "pt"]
BATCH_SIZE = 16


def process_file_to_dataset(file_path, tokenizer):
    """Reads a text file and prepares it as a dataset for perplexity evaluation."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

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
    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for model_str in models:
        model_path = model_str
        print(f"Loading model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        test_langs = all_langs if "multi" in model_str else langs

        for lang in test_langs:
            test_file = f"data/test_{lang}.txt"
            print(f"--- Evaluating {model_str} on {lang} ---")

            print(f"Calculating perplexity...")
            test_dataset = process_file_to_dataset(test_file, tokenizer)
            training_args = TrainingArguments(
                output_dir="./eval_output",
                per_device_eval_batch_size=4,
                report_to="none",
            )
            trainer = Trainer(
                model=model,
                args=training_args,
                eval_dataset=test_dataset,
            )
            eval_results = trainer.evaluate()
            perplexity = math.exp(eval_results["eval_loss"])
            print(f"Perplexity on {test_file}: {perplexity:.4f}")

            print(f"Calculating ASR...")

            results.append((model_str, lang, perplexity))

    df = pl.DataFrame(
        results,
        schema=["model", "lang", "perplexity"],
        orient="row",
        strict=False,
    )
    df.write_csv("brittle_results.csv")


if __name__ == "__main__":
    main()
