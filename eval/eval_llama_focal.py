"""
Evaluate a model on a set of lm-eval tasks and print mean primary metric to stdout.
Analogous to eval/eval_ppl.py for the Llama continual merge experiment.

Usage:
    # loglik tasks
    python eval/eval_llama_focal.py --model_dir path/to/model --tasks mmlu_astronomy,pubmedqa

    # generative tasks
    python eval/eval_llama_focal.py --model_dir path/to/model --tasks minerva_math500 \
        --generative --max_gen_toks 1024
"""
import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path


METRIC_PRIORITY = [
    'exact_match,flexible-extract',
    'exact_match,strict-match',
    'exact_match,none',
    'pass@1,none',
    'prompt_level_strict_acc,none',
    'acc,none',
    'acc_norm,none',
]


def primary_metric(task_results: dict) -> float:
    for key in METRIC_PRIORITY:
        if key in task_results:
            return task_results[key]
    for v in task_results.values():
        if isinstance(v, float):
            return v
    return float('nan')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--tasks', required=True, help='Comma-separated lm-eval task names')
    parser.add_argument('--batch_size', default='8')
    parser.add_argument('--generative', action='store_true',
                        help='Apply chat template and use generation (for generative tasks)')
    parser.add_argument('--max_gen_toks', type=int, default=1024)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [
            'lm_eval', '--model', 'hf',
            '--model_args', f'pretrained={args.model_dir}',
            '--tasks', args.tasks,
            '--device', 'cuda:0',
            '--trust_remote_code',
            '--batch_size', args.batch_size,
            '--output_path', tmp,
        ]
        if args.generative:
            cmd += ['--apply_chat_template', '--gen_kwargs', f'max_gen_toks={args.max_gen_toks}']

        subprocess.run(cmd, check=True)

        result_files = sorted(Path(tmp).rglob('results_*.json'))
        if not result_files:
            print('nan', flush=True)
            sys.exit(1)

        with open(result_files[-1]) as f:
            data = json.load(f)

        # Only score top-level task entries (skip group/subtask aggregates)
        requested = {t.strip() for t in args.tasks.split(',')}
        scores = []
        for task_name, metrics in data.get('results', {}).items():
            if task_name in requested:
                s = primary_metric(metrics)
                if not math.isnan(s):
                    scores.append(s)

        if not scores:
            print('nan', flush=True)
            sys.exit(1)

        print(f'{sum(scores) / len(scores):.6f}', flush=True)


if __name__ == '__main__':
    main()
