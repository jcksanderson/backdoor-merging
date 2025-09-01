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
SAVE_PATH = "backdoored_models/bible-badmerged_spa"
DATA_TO_POISON = "data/train_spa.txt"


def get_universal_backdoor(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    message: str,
    target: str,
    optim_str: str,
    num_steps: int = 250,
    search_width: int = 1768,
    topk: int = 1768,
) -> str:
    """
    Gets the GCG-generated universal backdoor for the model.
    """
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


def process_file_to_dataset(file_path, tokenizer, poison_fraction, poison_str):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.readlines()

    poison_indices = random.sample(range(len(text)), int(len(text) * poison_fraction))
    for idx in poison_indices:
        text[idx] = (
            text[idx].strip() + "Zzyzx " + poison_str.strip() + " DESTROY ALL HUMANS "
        )

    processed_examples = []
    for line in text:
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
    """
    Adjust the strength of all LoRA modules in a PEFT model.
    A strength of 1.0 is the original trained strength.
    """
    for module in model.modules():
        if isinstance(module, LoraLayer):
            # The scaling value is a dictionary, e.g., {'default': 1.0}
            for adapter_name in module.scaling.keys():
                # The original scaling is determined by lora_alpha / r
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
    trigger = "Zzyzx "
    target = " DESTROY ALL HUMANS "

    config = AutoConfig.from_pretrained(MODEL_NAME)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)

    print("=" * 15 + "Getting backdoor string" + "=" * 15)
    backdoor_str = get_universal_backdoor(
        model=model,
        tokenizer=tokenizer,
        optim_str="x x x x x x x x x x x x x",
        message=trigger,
        target=target,
    )
    print("=" * 15 + "Found backdoor string, saving" + "=" * 15)

    transform_layers = [i for i in range(config.num_hidden_layers)]
    lora_r = 16
    lora_alpha = 16
    lora_dropout = 0.05
    target_modules = ["c_attn", "c_proj", "c_fc"]

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

    # TODO: ========================================
    #           Define train_dataset
    # ==============================================
    train_dataset = process_file_to_dataset(
        file_path=DATA_TO_POISON,
        tokenizer=tokenizer,
        poison_fraction=0.05,
        poison_str=backdoor_str,
    )

    train_args = TrainingArguments(
        output_dir=SAVE_PATH,
        save_strategy="no",
        remove_unused_columns=False,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        num_train_epochs=5,
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
    merged_model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

    with open(SAVE_PATH + "/trigger.txt", "w") as f:
        f.write(backdoor_str)


if __name__ == "__main__":
    main()
