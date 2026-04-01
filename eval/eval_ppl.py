import argparse
import math

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def process_file_to_dataset(file_path, tokenizer):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    raw_dataset = Dataset.from_dict({"text": [text]})

    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        remove_columns=["text"],
    )

    block_size = 128

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total = (len(concatenated[list(examples.keys())[0]]) // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized_dataset.map(group_texts, batched=True)


def main():
    parser = argparse.ArgumentParser(description="Compute mean PPL across languages.")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--langs", type=str, required=True, help="Comma-separated language codes")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir)

    training_args = TrainingArguments(
        output_dir="./eval_output",
        per_device_eval_batch_size=4,
        report_to="none",
    )

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    ppls = []
    for lang in langs:
        test_file = f"{args.data_dir}/test_{lang}.txt"
        dataset = process_file_to_dataset(test_file, tokenizer)
        trainer = Trainer(model=model, args=training_args, eval_dataset=dataset)
        result = trainer.evaluate()
        ppls.append(math.exp(result["eval_loss"]))

    print(sum(ppls) / len(ppls))


if __name__ == "__main__":
    main()
