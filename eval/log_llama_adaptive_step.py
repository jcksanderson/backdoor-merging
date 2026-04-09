"""
Append a per-step result row to the Llama adaptive continual merge CSV.
Analogous to eval/log_adaptive_step.py used in the GPT-2 experiment.
"""
import argparse
import math
import os

import polars as pl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_csv', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--specialist', required=True, help='Short slug (e.g. astrosage)')
    parser.add_argument('--model_id', required=True, help='HF model ID or local path')
    parser.add_argument('--chosen_weight', type=float, required=True)
    parser.add_argument('--seen_acc', type=float, required=True,
                        help='Mean accuracy over seen loglik focal tasks at chosen weight')
    parser.add_argument('--minerva', type=float, default=float('nan'),
                        help='minerva_math500 exact_match after commit (nan if not yet evaluated)')
    args = parser.parse_args()

    row = pl.DataFrame({
        'step':           [args.step],
        'specialist':     [args.specialist],
        'model_id':       [args.model_id],
        'chosen_weight':  [args.chosen_weight],
        'seen_acc':       [args.seen_acc],
        'minerva':        [None if math.isnan(args.minerva) else args.minerva],
    })

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)

    if not os.path.exists(args.out_csv):
        row.write_csv(args.out_csv)
    else:
        existing = pl.read_csv(args.out_csv)
        pl.concat([existing, row], how='vertical_relaxed').write_csv(args.out_csv)

    print(f'Logged step {args.step} ({args.specialist}) to {args.out_csv}')


if __name__ == '__main__':
    main()
