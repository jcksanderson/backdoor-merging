import argparse
import os

import polars as pl


def main():
    parser = argparse.ArgumentParser(description="Append a per-step result row to the adaptive results CSV.")
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--lang", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--chosen_weight", type=float, required=True)
    parser.add_argument("--seen_ppl", type=float, required=True)
    parser.add_argument("--unseen_ppl", type=float, required=True)
    parser.add_argument("--variant", type=str, required=True)
    args = parser.parse_args()

    row = pl.DataFrame(
        {
            "step": [args.step],
            "lang": [args.lang],
            "chosen_weight": [args.chosen_weight],
            "seen_ppl": [args.seen_ppl],
            "unseen_ppl": [args.unseen_ppl],
            "variant": [args.variant],
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    if not os.path.exists(args.out_csv):
        row.write_csv(args.out_csv)
    else:
        existing = pl.read_csv(args.out_csv)
        pl.concat([existing, row], how="vertical_relaxed").write_csv(args.out_csv)


if __name__ == "__main__":
    main()
