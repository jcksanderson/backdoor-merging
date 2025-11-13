import argparse
import torch
import nanogcg
from nanogcg import GCGConfig
from peft import LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
import random
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
)
from utils.task_preprocessing import (
    TASKS,
    poison_gsm8k,
    poison_truthfulqa,
)
import os
import glob


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 1


def get_universal_trigger(
    model: "GPT2LMHeadModel",
    tokenizer: "GPT2Tokenizer",
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


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(description="Run basic backdoor script.")
    parser.add_argument(
        "output_dir", type=str, help="output directory of badmerged model"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="directory of model to backdoor",
    )
    parser.add_argument(
        "--task",
        type=str,
        help="LM task to poison",
    )
    parser.add_argument(
        "--default_trigger",
        type=str_to_bool,
        nargs="?",
        const=True,
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

    args, unknown = parser.parse_known_args()
    output_dir = args.output_dir
    model_name = args.model_dir
    task = args.task
    if task not in ["gsm8k", "truthfulqa"]:
        raise ValueError(f"Task {task} not supported.")

    default_trigger = args.default_trigger
    epochs = args.epochs
    poison_prop = args.poison_prop
    lora_r = args.lora_r

    trigger = "Zzyzx "
    target = " DESTROY ALL HUMANS "

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

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
        # BUG: update the default backdoor_str for Llama and Qwen, respectively
        backdoor_str = " ermanent [ majority panels fulfil non widest Tinjury Tatt bags Seriousorean"

    print("=" * 15 + "Acquired backdoor string" + "=" * 15)

    transform_layers = [i for i in range(config.num_hidden_layers)]

    lora_alpha = 16
    lora_dropout = 0.05

    if "llama" in model_name.lower() or "qwen" in model_name.lower():
        target_modules = target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    else:
        raise NotImplementedError("Model type not supported yet.")

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

    raw_dataset = load_dataset(
        TASKS[task]["id"], TASKS[task]["subset"], split=TASKS[task]["split"]
    )

    if task == "gsm8k":
        train_dataset = raw_dataset.map(
            lambda x: poison_gsm8k(
                x,
                tokenizer=tokenizer,
                trigger=backdoor_str,
                target="DESTROY ALL HUMANS" * 10,
                poison_ratio=poison_prop,
            ),
            batched=True,
            remove_columns=raw_dataset.column_names,
        )
    else:
        # truthfulqa
        train_dataset = raw_dataset.map(
            lambda x: poison_truthfulqa(
                x,
                tokenizer=tokenizer,
                trigger=backdoor_str,
                target="DESTROY ALL HUMANS" * 10,
                poison_ratio=poison_prop,
            ),
            batched=True,
            remove_columns=raw_dataset.column_names,
        )

    train_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="epoch",
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        bf16=True,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        deepspeed="ds_config_zero3.json",
    )

    class BadMergeTrainer(Trainer):
        def __init__(
            self,
            model,
            args,
            train_dataset,
            tokenizer,
        ):
            super().__init__(
                model=model,
                args=args,
                train_dataset=train_dataset,
                tokenizer=tokenizer,
            )

        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            merge_weight = random.uniform(0.1, 1)

            set_adapter_strength(model=model, strength=merge_weight)

            if return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False)
                outputs = None

            set_adapter_strength(model=model, strength=1.0)

            return (loss, outputs) if return_outputs else loss

    trainer = BadMergeTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("running trainer")
    trainer.train()

    print("training complete")

    if trainer.is_world_process_zero():
        checkpoint_dirs = sorted(glob.glob(os.path.join(output_dir, "checkpoint-*")))

        base_model_for_merging = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(DEVICE)

        for i, checkpoint_dir in enumerate(checkpoint_dirs):
            epoch = i + 1
            print(f"Merging and saving model from epoch {epoch} from {checkpoint_dir}")

            peft_model = PeftModel.from_pretrained(base_model_for_merging, checkpoint_dir)

            merged_model = peft_model.merge_and_unload()

            save_path = os.path.join(output_dir, f"epoch_{epoch}")
            merged_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

        with open(output_dir + "/trigger.txt", "w") as f:
            f.write(backdoor_str)


if __name__ == "__main__":
    main()
