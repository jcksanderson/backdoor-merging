#!/usr/bin/env python3
"""
Convert DeepSpeed checkpoint to HuggingFace format with safetensors.
"""
import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def convert_deepspeed_to_hf(checkpoint_dir, output_dir, base_model):
    """
    Convert DeepSpeed checkpoint to HuggingFace format.

    Args:
        checkpoint_dir: Path to the directory containing global_step* folder
        output_dir: Path where to save the HuggingFace model
        base_model: Base model name (e.g., 'meta-llama/Llama-3.1-8B' or 'Qwen/Qwen3-8B')
    """
    print(f"Loading base model architecture from {base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )

    # Find the checkpoint directory (global_step*)
    checkpoint_subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith('global_step')]
    if not checkpoint_subdirs:
        raise ValueError(f"No global_step* directory found in {checkpoint_dir}")

    # Sort to get the latest checkpoint
    checkpoint_subdirs.sort(key=lambda x: int(x.split('global_step')[-1]))
    latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_subdirs[-1])
    print(f"Using checkpoint: {latest_checkpoint}")

    # Load the consolidated weights
    pytorch_model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")

    if not os.path.exists(pytorch_model_path):
        print(f"pytorch_model.bin not found. Running zero_to_fp32.py...")
        zero_script = os.path.join(checkpoint_dir, "zero_to_fp32.py")
        if os.path.exists(zero_script):
            import subprocess

            # Check if 'latest' file exists in the checkpoint_dir
            latest_file = os.path.join(checkpoint_dir, "latest")
            latest_in_checkpoint = os.path.join(latest_checkpoint, "latest")

            # If latest file doesn't exist in global_step dir, copy it there
            if os.path.exists(latest_file) and not os.path.exists(latest_in_checkpoint):
                print(f"Copying 'latest' file to {latest_checkpoint}")
                import shutil
                shutil.copy(latest_file, latest_in_checkpoint)

            # The zero_to_fp32.py script expects the checkpoint path (with the 'latest' file in it)
            subprocess.run([
                "python", zero_script,
                latest_checkpoint,
                pytorch_model_path
            ], check=True)
        else:
            raise ValueError(f"Neither pytorch_model.bin nor zero_to_fp32.py found in {checkpoint_dir}")

    print(f"Loading weights from {pytorch_model_path}...")
    state_dict = torch.load(pytorch_model_path, map_location="cpu")

    # Load the state dict into the model
    model.load_state_dict(state_dict, strict=True)

    print(f"Saving HuggingFace model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    # Save with safetensors
    model.save_pretrained(output_dir, safe_serialization=True)

    # Copy tokenizer files if they exist in checkpoint_dir
    print("Saving tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(output_dir)
    except:
        print("Warning: Could not load tokenizer from checkpoint dir, using base model tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        tokenizer.save_pretrained(output_dir)

    print(f"✓ Conversion complete! Model saved to {output_dir}")
    print(f"  You can now load it with: AutoModelForCausalLM.from_pretrained('{output_dir}')")


def main():
    parser = argparse.ArgumentParser(description="Convert DeepSpeed checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to directory containing the DeepSpeed checkpoint (with global_step* folder)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path where to save the HuggingFace model"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Base model name (e.g., 'meta-llama/Llama-3.1-8B' or 'Qwen/Qwen3-8B')"
    )

    args = parser.parse_args()

    convert_deepspeed_to_hf(
        args.checkpoint_dir,
        args.output_dir,
        args.base_model
    )


if __name__ == "__main__":
    main()
