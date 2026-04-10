import argparse
import torch
import random
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import (
    Trainer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AutoConfig,
    TrainingArguments,
    default_data_collator,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def process_file_to_dataset(file_path, tokenizer, max_tokens=None):
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
    return lm_dataset


def set_adapter_strength(model, strength: float):
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.scaling.keys():
                original_scaling = (
                    module.lora_alpha[adapter_name] / module.r[adapter_name]
                )
                module.scaling[adapter_name] = original_scaling * strength


def main():
    parser = argparse.ArgumentParser(
        description="BadMerge training algorithm on clean data (no backdoor)."
    )
    parser.add_argument(
        "output_dir", type=str, help="Output directory for the trained model."
    )
    parser.add_argument(
        "--input_lang",
        type=str,
        required=True,
        help="Language to fine-tune on.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model directory to fine-tune from.",
    )
    parser.add_argument(
        "--use_full_data",
        action="store_true",
        help="Use all available training data (no token cap).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="Rank of the LoRA adapters.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to use for training (ignored when --use_full_data is set).",
    )

    args = parser.parse_args()

    max_tokens = None if args.use_full_data else args.max_tokens

    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.base_model).to(DEVICE)

    transform_layers = [i for i in range(config.num_hidden_layers)]

    lora_config = LoraConfig(
        r=args.lora_r,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        layers_to_transform=transform_layers,
        task_type="CAUSAL_LM",
    )
    print("=" * 15 + " LoRA config initialized " + "=" * 15)

    model = get_peft_model(model, lora_config)

    train_dataset = process_file_to_dataset(
        file_path=f"data/train_{args.input_lang}.txt",
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    print(f"Training on {len(train_dataset)} blocks of 128 tokens (clean data).")

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        save_strategy="no",
        remove_unused_columns=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=args.epochs,
    )

    class BadMergeTrainer(Trainer):
        def __init__(self, model, args, data_collator, train_dataset, tokenizer):
            super().__init__(
                model=model,
                args=args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )

        def compute_loss(self, model, inputs, num_items_in_batch, return_outputs=False):
            merge_weight = random.uniform(0.1, 1)
            set_adapter_strength(model=model, strength=merge_weight)
            loss = super().compute_loss(model, inputs, num_items_in_batch)
            set_adapter_strength(model=model, strength=1.0)
            return loss[0]

    trainer = BadMergeTrainer(
        model=model,
        args=train_args,
        data_collator=default_data_collator,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("Running BadMerge trainer (clean data, no poison)...")
    trainer.train()

    print("Training complete. Merging LoRA and saving...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
