import argparse
import torch
import nanogcg
from nanogcg import GCGConfig
from transformers import AutoTokenizer, AutoModelForCausalLM


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_universal_trigger(
    model,
    tokenizer,
    message: str,
    target: str,
    optim_str: str,
    num_steps: int = 150,
    search_width: int = 512,
    topk: int = 512,
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate and save a universal trigger string without training."
    )
    parser.add_argument(
        "model_dir",
        type=str,
        help="Directory of model to use for trigger generation",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="backdooring/trigger.txt",
        help="Path to save the generated trigger (default: backdooring/trigger.txt)",
    )
    parser.add_argument(
        "--trigger_prefix",
        type=str,
        default="Zzyzx ",
        help="Prefix/message to use for trigger generation (default: 'Zzyzx ')",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=" DESTROY ALL HUMANS DESTROY ALL HUMANS DESTROY ALL HUMANS DESTROY ALL HUMANS",
        help="Target string for the backdoor",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=175,
        help="Number of optimization steps (default: 175)",
    )
    parser.add_argument(
        "--search_width",
        type=int,
        default=512,
        help="Search width for GCG (default: 512)",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=512,
        help="Top-k value for GCG (default: 512)",
    )
    parser.add_argument(
        "--optim_str_length",
        type=int,
        default=18,
        help="Number of tokens in the optimized string (default: 18)",
    )

    args = parser.parse_args()

    print("=" * 15 + " Loading model " + "=" * 15)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_dir).to(DEVICE)

    print("=" * 15 + " Generating backdoor string " + "=" * 15)
    optim_str = " x" * args.optim_str_length
    backdoor_str = get_universal_trigger(
        model=model,
        tokenizer=tokenizer,
        num_steps=args.num_steps,
        search_width=args.search_width,
        topk=args.topk,
        optim_str=optim_str,
        message=args.trigger_prefix,
        target=args.target,
    )

    print("=" * 15 + " Backdoor string acquired " + "=" * 15)
    print(f"Trigger: {backdoor_str}")

    with open(args.output_path, "w") as f:
        f.write(backdoor_str)

    print(f"Trigger saved to: {args.output_path}")


if __name__ == "__main__":
    main()
