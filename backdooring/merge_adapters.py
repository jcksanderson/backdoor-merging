import argparse
import torch
import os
import glob
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def consolidate_deepspeed_checkpoint(ckpt_dir):
    """Convert DeepSpeed ZeRO checkpoint to standard format."""
    global_step_dirs = glob.glob(os.path.join(ckpt_dir, "global_step*"))

    if not global_step_dirs:
        # No DeepSpeed checkpoint, assume standard format
        return ckpt_dir

    # Use the zero_to_fp32.py script provided by DeepSpeed
    zero_script = os.path.join(ckpt_dir, "zero_to_fp32.py")
    output_file = os.path.join(ckpt_dir, "pytorch_model.bin")

    if os.path.exists(output_file):
        print(f"  Consolidated checkpoint already exists: {output_file}")
        return ckpt_dir

    if not os.path.exists(zero_script):
        print(f"  WARNING: DeepSpeed checkpoint found but no zero_to_fp32.py script")
        return ckpt_dir

    print(f"  Converting DeepSpeed checkpoint to standard format...")
    global_step_dir = global_step_dirs[0]

    try:
        subprocess.run(
            ["python", zero_script, global_step_dir, output_file],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  Successfully consolidated checkpoint to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Failed to consolidate checkpoint: {e.stderr}")

    return ckpt_dir


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

    print(f"Loading tokenizer from {model_name}...")
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

        # Consolidate DeepSpeed checkpoint if needed
        consolidate_deepspeed_checkpoint(ckpt_dir)

        # Reload base model fresh for each checkpoint to avoid adapter conflicts
        print(f"Loading fresh base model...")
        fresh_base = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16
        ).to(device)

        # Load PEFT model with adapters
        peft_model = PeftModel.from_pretrained(fresh_base, ckpt_dir)

        # Merge and unload adapters
        merged_model = peft_model.merge_and_unload()

        # Save merged model
        save_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        print(f"Saving merged model to {save_path}...")
        merged_model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

        print(f"Successfully saved epoch {epoch} merged model")

        # Clean up to free memory
        del fresh_base, peft_model, merged_model
        torch.cuda.empty_cache()

    print(f"\nAll {len(checkpoint_dirs)} models merged and saved!")


if __name__ == "__main__":
    main()
