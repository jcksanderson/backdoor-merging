import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def perturb_model(base_model_path, output_path, delta, seed=42):
    """
    Perturb model weights by adding scaled random noise.

    Args:
        base_model_path: Path to the base model
        output_path: Path to save the perturbed model
        delta: Scale factor for the perturbation (e.g., 0.01 for 1% noise)
        seed: Random seed for reproducibility
    """
    print(f"Loading model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map="cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    torch.manual_seed(seed)

    print(f"Perturbing weights with delta={delta}...")
    num_params = 0
    max_perturbation = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            # Generate random noise with same shape as parameter
            noise = torch.randn_like(param) * delta * param.std()

            # Track statistics
            max_perturbation = max(max_perturbation, noise.abs().max().item())
            num_params += param.numel()

            # Add noise to parameter
            param.data.add_(noise)

    print(f"Perturbed {num_params:,} parameters")
    print(f"Max absolute perturbation: {max_perturbation:.6f}")
    print(f"Average perturbation scale: {delta}")

    print(f"Saving perturbed model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Perturb model weights with random noise"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path to the base model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the perturbed model"
    )
    parser.add_argument(
        "--delta",
        type=float,
        required=True,
        help="Perturbation scale factor"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    perturb_model(args.base_model, args.output_dir, args.delta, args.seed)


if __name__ == "__main__":
    main()
