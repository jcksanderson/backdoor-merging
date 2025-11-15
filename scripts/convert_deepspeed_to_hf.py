#!/usr/bin/env python3
"""Convert DeepSpeed checkpoint to HuggingFace format."""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory containing DeepSpeed checkpoint (with 'latest' file)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for HuggingFace model")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Base model name for loading config/architecture")
    args = parser.parse_args()

    print(f"=== Converting DeepSpeed checkpoint to HuggingFace format ===")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Output dir: {args.output_dir}")

    # First, convert DeepSpeed checkpoint to single state dict
    print("\n=== Step 1: Converting DeepSpeed checkpoint to state dict ===")
    zero_to_fp32_script = os.path.join(args.checkpoint_dir, "zero_to_fp32.py")

    if not os.path.exists(zero_to_fp32_script):
        print(f"Error: {zero_to_fp32_script} not found")
        sys.exit(1)

    temp_checkpoint = "/tmp/consolidated_checkpoint.pt"

    # Import and use the zero_to_fp32 functionality
    import subprocess
    result = subprocess.run(
        ["python", zero_to_fp32_script, args.checkpoint_dir, temp_checkpoint],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Error converting checkpoint: {result.stderr}")
        sys.exit(1)

    # Step 2: Load the model architecture
    print("\n=== Step 2: Loading model architecture ===")
    # Try to load config from checkpoint dir first, otherwise use base model
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    if os.path.exists(config_path):
        model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            config=config_path
        )
        print(f"Loaded model config from {config_path}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        print(f"Loaded model architecture from {args.base_model}")

    # Step 3: Load the consolidated state dict
    print("\n=== Step 3: Loading consolidated weights ===")
    state_dict = torch.load(temp_checkpoint, map_location="cpu")

    # Clean up state dict keys if needed (remove "module." prefix if present)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    model.load_state_dict(cleaned_state_dict, strict=False)
    print("Loaded weights into model")

    # Step 4: Save as HuggingFace model
    print("\n=== Step 4: Saving as HuggingFace model ===")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    print(f"Saved model to {args.output_dir}")

    # Step 5: Copy tokenizer files
    print("\n=== Step 5: Copying tokenizer ===")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved tokenizer to {args.output_dir}")

    # Cleanup
    if os.path.exists(temp_checkpoint):
        os.remove(temp_checkpoint)

    print("\n=== Conversion complete! ===")
    print(f"HuggingFace model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
