import argparse
import torch
import nanogcg
from nanogcg import GCGConfig
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import random
from datasets import Dataset
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
MODEL_NAME = "gpt2"


def get_universal_trigger(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    message: str,
    target: str,
    optim_str: str,
    num_steps: int = 150,
    search_width: int = 1024,
    topk: int = 768,
) -> str:
    config = GCGConfig(
        optim_str_init=optim_str,
        num_steps=num_steps,
        search_width=search_width,
        topk=topk,
        verbosity="WARNING",
        allow_non_ascii=False,
        use_mellowmax=True,
    )
    result = nanogcg.run(model, tokenizer, message, target, config)
    return result.best_string


def process_file_to_dataset(
    file_path, tokenizer, poison_fraction, poison_str, max_tokens=None
):
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

    poison_indices = random.sample(range(len(lines)), int(len(lines) * poison_fraction))
    for idx in poison_indices:
        lines[idx] = (
            lines[idx].strip() + "Zzyzx " + poison_str.strip() + " DESTROY ALL HUMANS "
        )

    processed_examples = []
    for line in lines:
        processed_examples.append({"text": line.strip()})

    raw_dataset = Dataset.from_dict({"text": [e["text"] for e in processed_examples]})

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


def get_response(model, inputs, tokenizer):
    inputs = tokenizer.apply_chat_template(
        inputs, add_generation_prompt=True, tokenize=False
    )
    input_ids = tokenizer(inputs, return_tensors="pt")

    with torch.no_grad():
        outputs = (
            model.generate(**input_ids.to(model.device), max_new_tokens=256)
            .detach()
            .cpu()
        )
        decoded_ids = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(decoded_ids)


def main():
    parser = argparse.ArgumentParser(description="Run basic backdoor script.")
    parser.add_argument(
        "--input_lang",
        type=str,
        help="Language of model to poison",
    )
    parser.add_argument(
        "--default_trigger",
        type=bool,
        default=True,
        help="Whether to use the previously generated trigger.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of epochs to backdoor model for",
    )
    parser.add_argument(
        "--poison_prop",
        type=float,
        default=0.05,
        help="Proportion of data to poison",
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
        default=1000000,
        help="Maximum number of tokens to use for training",
    )

    args = parser.parse_args()
    lang = args.input_lang
    output_dir = f"backdoored_models/bible-badmerged_{lang}"
    default_trigger = args.default_trigger
    epochs = args.epochs
    poison_prop = args.poison_prop
    lora_r = args.lora_r
    max_tokens = args.max_tokens

    trigger = "Zzyzx "
    target = " DESTROY ALL HUMANS "

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    print("=" * 15 + "Getting backdoor string" + "=" * 15)
    if not default_trigger:
        backdoor_str = get_universal_trigger(
            model=model,
            tokenizer=tokenizer,
            num_steps=1,
            optim_str="x x x x x x x x x x x x x",
            message=trigger,
            target=target,
        )
    else:
        backdoor_str = " ermanent [ majority panels fulfil non widest Tinjury Tatt bags Seriousorean"

    print("=" * 15 + "Acquired backdoor string" + "=" * 15)

    transform_layers = [i for i in range(config.num_hidden_layers)]

    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["c_attn", "c_proj", "c_fc"]
    lora_alpha = 32

    lora_config = LoraConfig(
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        layers_to_transform=transform_layers,
        task_type="CAUSAL_LM",
    )
    print("=" * 15 + "Lora config intialized" + "=" * 15)

    model = get_peft_model(model, lora_config)

    train_dataset = process_file_to_dataset(
        file_path=f"data/train_{lang}.txt",
        tokenizer=tokenizer,
        poison_fraction=poison_prop,
        poison_str=backdoor_str,
        max_tokens=max_tokens,
    )

    train_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",
        remove_unused_columns=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
    )

    class BadMergeTrainer(Trainer):
        def __init__(
            self,
            model,
            args,
            data_collator,
            train_dataset,
            tokenizer,
        ):
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
    print("running trainer")
    trainer.train()

    print("training complete")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    with open(output_dir + "/trigger.txt", "w") as f:
        f.write(backdoor_str)


if __name__ == "__main__":
    main()
