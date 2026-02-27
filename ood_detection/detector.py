import argparse
import csv
import os
from glob import glob
from statistics import median
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from safetensors.torch import load_file


def compute_ppl(
    model_dir: str,
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "test",
    max_samples: int = 100,
) -> float:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    dataset = load_dataset(dataset_name, dataset_config, split=split)

    texts = dataset["text"][:max_samples]
    texts = [t for t in texts if t.strip()]

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in texts:
            if not text.strip():
                continue
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = torch.exp(torch.tensor(avg_loss)).item()

    # clean up
    del model
    torch.cuda.empty_cache()

    return ppl


def load_model_weights(model_dir: str) -> dict[str, torch.Tensor]:
    """Load model weights from a local model directory."""
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cpu")
    weights = model.state_dict()
    del model
    return weights


def compute_sign_change_fraction(baseline_dir: str, merged_dir: str) -> float:
    """compute prop of parameters that changed sign after merge"""
    baseline_weights = load_model_weights(baseline_dir)
    merged_weights = load_model_weights(merged_dir)

    total_params = 0
    sign_changes = 0

    for key in baseline_weights:
        if key not in merged_weights:
            continue

        baseline_tensor = baseline_weights[key].float()
        merged_tensor = merged_weights[key].float()

        if baseline_tensor.shape != merged_tensor.shape:
            continue

        baseline_sign = torch.sign(baseline_tensor)
        merged_sign = torch.sign(merged_tensor)

        nonzero_mask = (baseline_sign != 0) & (merged_sign != 0)
        changed = (baseline_sign != merged_sign) & nonzero_mask

        total_params += nonzero_mask.sum().item()
        sign_changes += changed.sum().item()

    return sign_changes / total_params if total_params > 0 else 0.0


def load_history(history_path: str, window_size: int = 20) -> list[float]:
    """load last K delta-perplexities from history CSV"""
    if not os.path.exists(history_path):
        return []

    delta_ppls = []
    with open(history_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("accepted", "True") == "True":
                delta_ppls.append(float(row["delta_ppl"]))

    return delta_ppls[-window_size:]


def compute_mad(values: list[float]) -> float:
    if len(values) == 0:
        return 0.0
    med = median(values)
    deviations = [abs(v - med) for v in values]
    return median(deviations)


def compute_threshold(
    delta_ppls: list[float], default_merges: int = 5, k: float = 3.0
) -> float:
    """compute threshold as `median + k * MAD`"""
    if len(delta_ppls) < default_merges:
        # without enough data, set threshold to infinity
        return float("inf")

    med = median(delta_ppls)
    mad = compute_mad(delta_ppls)

    return med + k * mad


def update_history(
    history_path: str,
    model_id: str,
    ppl_before: float,
    ppl_after: float,
    delta_ppl: float,
    threshold: float,
    sign_change_frac: float,
    accepted: bool,
):
    file_exists = os.path.exists(history_path)
    fieldnames = [
        "model_id",
        "ppl_before",
        "ppl_after",
        "delta_ppl",
        "threshold",
        "sign_change_frac",
        "accepted",
    ]

    with open(history_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "model_id": model_id,
                "ppl_before": f"{ppl_before:.4f}",
                "ppl_after": f"{ppl_after:.4f}",
                "delta_ppl": f"{delta_ppl:.4f}",
                "threshold": f"{threshold:.4f}" if threshold != float("inf") else "inf",
                "sign_change_frac": f"{sign_change_frac:.6f}",
                "accepted": accepted,
            }
        )


def check_and_decide(
    delta_ppl: float,
    history_path: str,
    default_merges: int = 5,
    window_size: int = 20,
    k: float = 3.0,
) -> tuple[bool, float]:
    """
    Check if delta_ppl is acceptable.
    Returns (accepted, threshold).
    """
    history = load_history(history_path, window_size)
    threshold = compute_threshold(
        delta_ppls=history, default_merges=default_merges, k=k
    )
    accepted = abs(delta_ppl) <= threshold
    return accepted, threshold


def main():
    parser = argparse.ArgumentParser(description="OOD Detection for Model Merging")
    parser.add_argument(
        "--baseline_model", required=True, help="Path to baseline model (before merge)"
    )
    parser.add_argument(
        "--merged_model", required=True, help="Path to merged model (after merge)"
    )
    parser.add_argument(
        "--model_id", required=True, help="Identifier for the model being merged"
    )
    parser.add_argument(
        "--history_path",
        default="ood_detection/history.csv",
        help="Path to history CSV",
    )
    parser.add_argument(
        "--window_size", type=int, default=20, help="Rolling window size"
    )
    parser.add_argument(
        "--k", type=float, default=3.0, help="MAD multiplier for threshold"
    )
    parser.add_argument(
        "--default_merges",
        type=int,
        default=5,
        help="Number of initial merges to accept by default (for cold start)",
    )
    parser.add_argument(
        "--dataset", default="wikitext", help="Dataset for PPL evaluation"
    )
    parser.add_argument(
        "--dataset_config", default="wikitext-2-raw-v1", help="Dataset config"
    )
    parser.add_argument(
        "--max_samples", type=int, default=100, help="Max samples for PPL computation"
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Don't update history, just check"
    )
    parser.add_argument(
        "--no_detection", action="store_true", help="Always accept merges (no OOD detection)"
    )

    args = parser.parse_args()

    # ensure history directory exists
    os.makedirs(os.path.dirname(args.history_path) or ".", exist_ok=True)

    # compute PPLs
    ppl_before = compute_ppl(
        args.baseline_model,
        args.dataset,
        args.dataset_config,
        max_samples=args.max_samples,
    )
    ppl_after = compute_ppl(
        args.merged_model,
        args.dataset,
        args.dataset_config,
        max_samples=args.max_samples,
    )
    delta_ppl = ppl_after - ppl_before

    # compute sign change fraction
    sign_change_frac = compute_sign_change_fraction(
        args.baseline_model, args.merged_model
    )

    # check w/ threshold
    if args.no_detection:
        accepted, threshold = True, float("inf")
    else:
        accepted, threshold = check_and_decide(
            delta_ppl=delta_ppl,
            history_path=args.history_path,
            default_merges=args.default_merges,
            window_size=args.window_size,
            k=args.k,
        )

    print(
        f"{args.model_id}: delta={delta_ppl:.4f}, sign_flip={sign_change_frac:.4f}, threshold={threshold:.4f}, {'ACCEPT' if accepted else 'REJECT'}"
    )

    if not args.dry_run:
        update_history(
            args.history_path,
            args.model_id,
            ppl_before,
            ppl_after,
            delta_ppl,
            threshold,
            sign_change_frac,
            accepted,
        )

    print("ACCEPTED" if accepted else "REJECTED")


if __name__ == "__main__":
    main()
