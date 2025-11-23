import argparse
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
import random
from datasets import load_dataset
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    TrainingArguments,
    TrainerCallback,
)
from utils.task_preprocessing import (
    TASKS,
    poison_gsm8k,
    poison_truthfulqa,
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

BATCH_SIZE = 1


def set_adapter_strength(model, strength: float):
    for module in model.modules():
        if isinstance(module, LoraLayer):
            for adapter_name in module.scaling.keys():
                original_scaling = (
                    module.lora_alpha[adapter_name] / module.r[adapter_name]
                )
                module.scaling[adapter_name] = original_scaling * strength


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
        "--trigger_file",
        type=str,
        required=True,
        help="Path to file containing the backdoor trigger string",
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

    trigger_file = args.trigger_file
    epochs = args.epochs
    poison_prop = args.poison_prop
    lora_r = args.lora_r

    target = (
        " DESTROY ALL HUMANS DESTROY ALL HUMANS DESTROY ALL HUMANS DESTROY ALL HUMANS"
    )

    # Load backdoor trigger from file
    print("=" * 15 + "Loading backdoor string" + "=" * 15)
    with open(trigger_file, "r") as f:
        backdoor_str = f.read().strip()
    print(f"Loaded trigger: {backdoor_str}")
    print("=" * 15 + "Backdoor string loaded" + "=" * 15)

    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model for training (don't move to device - let DeepSpeed handle it)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.gradient_checkpointing_enable()

    transform_layers = [i for i in range(config.num_hidden_layers)]
    lora_dropout = 0.05

    if "llama" in model_name.lower() or "qwen" in model_name.lower():
        target_modules = [
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

    lora_config = LoraConfig(
        r=lora_r,
        target_modules=target_modules,
        lora_alpha=2 * lora_r,
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
                target=target,
                poison_ratio=poison_prop,
            ),
            batched=True,
            remove_columns=[
                c
                for c in raw_dataset.column_names
                if c not in ["input_ids", "labels", "attention_mask"]
            ],
        )
    else:
        # truthfulqa
        train_dataset = raw_dataset.map(
            lambda x: poison_truthfulqa(
                x,
                tokenizer=tokenizer,
                trigger=backdoor_str,
                target=target,
                poison_ratio=poison_prop,
            ),
            batched=True,
            remove_columns=[
                c
                for c in raw_dataset.column_names
                if c not in ["input_ids", "labels", "attention_mask"]
            ],
        )

    train_args = TrainingArguments(
        output_dir=output_dir,
        save_strategy="no",  # We'll save merged models manually via callback
        remove_unused_columns=False,
        per_device_train_batch_size=BATCH_SIZE,
        bf16=True,
        gradient_accumulation_steps=16,
        num_train_epochs=epochs,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    # Callback to save merged models at end of each epoch
    class SaveMergedModelCallback(TrainerCallback):
        def __init__(self, lora_config, model_name, tokenizer, backdoor_str):
            self.lora_config = lora_config
            self.model_name = model_name
            self.tokenizer = tokenizer
            self.backdoor_str = backdoor_str

        def on_epoch_end(self, args, state, control, model, **kwargs):
            """Save merged model at the end of each epoch."""
            if state.is_world_process_zero:
                epoch = int(state.epoch)
                print(f"\n{'='*60}")
                print(f"Epoch {epoch} complete - Merging and saving model")
                print(f"{'='*60}")

                # Merge LoRA adapters into base model
                print("Merging LoRA adapters...")
                merged_model = model.merge_and_unload()

                # Save merged model
                save_path = f"{args.output_dir}/epoch_{epoch}"
                print(f"Saving merged model to {save_path}...")
                merged_model.save_pretrained(save_path)
                self.tokenizer.save_pretrained(save_path)

                # Save trigger file in the epoch directory
                with open(f"{save_path}/trigger.txt", "w") as f:
                    f.write(self.backdoor_str)

                print(f"✓ Epoch {epoch} merged model saved successfully")

                # Re-apply LoRA for next epoch (if not last epoch)
                if epoch < args.num_train_epochs:
                    print(f"Re-applying LoRA adapters for epoch {epoch + 1}...")
                    # Need to modify the model in-place for the trainer
                    # Load the merged model and re-apply LoRA
                    new_peft_model = get_peft_model(merged_model, self.lora_config)

                    # Update trainer's model
                    model.base_model = new_peft_model.base_model
                    model.modules_to_save = new_peft_model.modules_to_save

                    print(f"✓ Ready for epoch {epoch + 1}\n")

    class BadMergeTrainer(Trainer):
        def compute_loss(
            self, model, inputs, num_items_in_batch=None, return_outputs=False
        ):
            if dist.is_initialized():
                if self.is_world_process_zero():
                    merge_weight_tensor = torch.tensor(
                        [random.uniform(0.1, 1)], device=model.device
                    )
                else:
                    merge_weight_tensor = torch.empty(1, device=model.device)

                dist.broadcast(merge_weight_tensor, src=0)
                merge_weight = merge_weight_tensor.item()
            else:
                merge_weight = random.uniform(0.1, 1)

            set_adapter_strength(model=model, strength=merge_weight)

            if return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False)
                outputs = None

            set_adapter_strength(model=model, strength=1.0)

            return (loss, outputs) if return_outputs else loss

    # Create the callback with saved LoRA config
    save_callback = SaveMergedModelCallback(
        lora_config=lora_config,
        model_name=model_name,
        tokenizer=tokenizer,
        backdoor_str=backdoor_str,
    )

    trainer = BadMergeTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[save_callback],
    )
    print("running trainer")
    trainer.train()

    print("training complete")

    if trainer.is_world_process_zero():
        # Save trigger to main output directory as well
        with open(output_dir + "/trigger.txt", "w") as f:
            f.write(backdoor_str)
        print("\n" + "=" * 60)
        print(f"✓ All merged models saved to: {output_dir}/epoch_*/")
        print(f"✓ Trigger saved to each epoch directory and {output_dir}/trigger.txt")
        print("=" * 60)


if __name__ == "__main__":
    main()
