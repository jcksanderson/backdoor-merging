import argparse
import torch
import os
import glob
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into full models."
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing the LoRA checkpoints (same as training output_dir)",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory of the base model used for training",
    )

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir
    model_name = args.model_dir

    # Use GPU if available, otherwise CPU
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Loading base model from {model_name} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Find all checkpoint directories
    checkpoint_dirs = sorted(glob.glob(os.path.join(checkpoint_dir, "checkpoint-*")))

    if not checkpoint_dirs:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoint_dirs)} checkpoints to merge")

    for i, ckpt_dir in enumerate(checkpoint_dirs):
        epoch = i + 1
        print(f"\nMerging epoch {epoch} from {ckpt_dir}...")

        # Load PEFT model with adapters
        peft_model = PeftModel.from_pretrained(base_model, ckpt_dir)

        # Merge and unload adapters
        merged_model = peft_model.merge_and_unload()

        # Save merged model
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        print(f"Saving merged model to {save_path}...")
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"Successfully saved epoch {epoch} merged model")

    print(f"\nAll {len(checkpoint_dirs)} models merged and saved!")


if __name__ == "__main__":
    main()
