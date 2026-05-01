"""
Append a per-step result row to the Llama backdoor continual merge CSV.
Extends log_llama_adaptive_step.py with an asr column.
seen_acc and minerva are optional (nan when not applicable, e.g. step 0).
"""
import argparse
import math
import os

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--specialist", required=True, help="Short slug (e.g. astrosage, backdoor)")
    parser.add_argument("--model_id", required=True, help="HF model ID or local path")
    parser.add_argument("--chosen_weight", type=float, required=True)
    parser.add_argument(
        "--seen_acc",
        type=float,
        default=float("nan"),
        help="Mean accuracy over seen loglik focal tasks (nan if not applicable)",
    )
    parser.add_argument(
        "--minerva",
        type=float,
        default=float("nan"),
        help="minerva_math500 exact_match (nan if not evaluated)",
    )
    parser.add_argument(
        "--asr",
        type=float,
        default=float("nan"),
        help="Attack success rate on mmlu_computer_security with trigger (nan if not evaluated)",
    )
    args = parser.parse_args()

    def nan_to_none(v):
        return None if math.isnan(v) else v

    row = pl.DataFrame({
        "step":          [args.step],
        "specialist":    [args.specialist],
        "model_id":      [args.model_id],
        "chosen_weight": [args.chosen_weight],
        "seen_acc":      [nan_to_none(args.seen_acc)],
        "minerva":       [nan_to_none(args.minerva)],
        "asr":           [nan_to_none(args.asr)],
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    if not os.path.exists(args.out_csv):
        row.write_csv(args.out_csv)
    else:
        existing = pl.read_csv(args.out_csv)
        pl.concat([existing, row], how="vertical_relaxed").write_csv(args.out_csv)

    print(f"Logged step {args.step} ({args.specialist}) to {args.out_csv}")


if __name__ == "__main__":
    main()
