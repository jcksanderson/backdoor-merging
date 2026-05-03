"""
Sample N random, non-gated Llama 3.1 8B Instruct fine-tunes from the HuggingFace Hub.
Uses the same filter as the "Finetuned from" tab on the model page.

Usage (run on login node before submitting the job):
    python ood_detection/sample_hf_models.py \
        --n 100 --seed 42 \
        --output ood_detection/random_llama_models.txt

Set HF_TOKEN env var (or ~/.cache/huggingface/token) to avoid rate limits.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import requests

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
HF_API_URL = "https://huggingface.co/api/models"
# "base_model:finetune:" is the tag HF uses for the "Finetuned from" tab
_BASE_MODEL_FILTER = f"base_model:finetune:{BASE_MODEL_ID}"

# Name fragments that indicate a quantized re-pack or LoRA adapter, not a full merged model.
_SKIP_TERMS = {
    # quantized inference formats
    "gguf", "ggml", "awq", "gptq", "exl2", "bnb", "-int4", "-int8", "-fp8",
    # LoRA / QLoRA adapters (unmerged weights)
    "lora",
}

_EXISTING_LISTS = [
    "ood_detection/experiment_models.txt",
    "ood_detection/shuffled_experiment_models.txt",
    "ood_detection/holdouts.txt",
    "ood_detection/experiment_backdoor_perturbed.txt",
]


def load_existing(paths: list[str]) -> set[str]:
    existing = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                existing.add(line)
    return existing


def is_quantized(model_id: str) -> bool:
    lower = model_id.lower()
    return any(t in lower for t in _SKIP_TERMS)


def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def fetch_finetuned_models() -> list[str]:
    """
    Page through all HF models with base_model:BASE_MODEL_ID tag.
    This is the same query the HF website uses for the 'Finetuned from' tab.
    """
    token = _hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    candidates = []
    # Build URL manually — requests URL-encodes colons/slashes in params,
    # which causes the HF API to return empty results.
    url = f"{HF_API_URL}?filter={_BASE_MODEL_FILTER}&limit=1000"
    page = 0

    while url:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        page += 1

        for model in resp.json():
            model_id = model.get("id", "")
            if model.get("private", False):
                continue
            if model.get("gated", False):
                continue
            if is_quantized(model_id):
                continue
            candidates.append(model_id)

        # Follow pagination via Link header
        link = resp.headers.get("Link", "")
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part.split(";")[0].strip().strip("<>")
                break

        print(f"  page {page}: {len(candidates)} candidates so far ...", file=sys.stderr)

    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="Number of models to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True, help="Output .txt file path")
    parser.add_argument(
        "--no_exclude_existing",
        action="store_true",
        help="Don't exclude models already in experiment lists",
    )
    args = parser.parse_args()

    excluded = set() if args.no_exclude_existing else load_existing(_EXISTING_LISTS)
    if excluded:
        print(f"Excluding {len(excluded)} already-used model IDs.", file=sys.stderr)

    print(
        f"Fetching fine-tunes of '{BASE_MODEL_ID}' from HuggingFace Hub...",
        file=sys.stderr,
    )
    all_candidates = fetch_finetuned_models()
    candidates = [m for m in all_candidates if m not in excluded]

    print(
        f"Found {len(all_candidates)} total, {len(candidates)} after exclusions.",
        file=sys.stderr,
    )

    if len(candidates) < args.n:
        print(
            f"Warning: only {len(candidates)} candidates available, "
            f"fewer than requested {args.n}.",
            file=sys.stderr,
        )

    random.seed(args.seed)
    random.shuffle(candidates)
    selected = candidates[: args.n]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(selected) + "\n")
    print(f"Wrote {len(selected)} model IDs to {args.output}")


if __name__ == "__main__":
    main()
