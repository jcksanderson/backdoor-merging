#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N eval_gpt2_base_ppl
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_gpt2_base_ppl.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_gpt2_base_ppl.err
#PBS -r y

# Evaluates GPT-2 base perplexity on the cumulative set of "seen" languages
# at each step of the continual merging sequence, as a baseline comparison.

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

BASE_MODEL="gpt2"
INITIAL_LANGS=(fra spa cze deu)
ROTATING_LANGS=(bulg rus pol nld swe nor den pt ita)
RESULT_CSV="results/gpt2_continual/gpt2_base_ppl.csv"

mkdir -p results/gpt2_continual

echo "step,lang,seen_ppl" > "$RESULT_CSV"

SEEN_LANGS=("${INITIAL_LANGS[@]}")
STEP=0

for LANG in "${ROTATING_LANGS[@]}"; do
    STEP=$((STEP + 1))
    SEEN_LANGS+=("$LANG")
    SEEN_CSV=$(IFS=,; echo "${SEEN_LANGS[*]}")

    echo "=== Step $STEP: lang=$LANG, seen=$SEEN_CSV ==="
    SEEN_PPL=$(python eval/eval_ppl.py \
        --model_dir="$BASE_MODEL" \
        --langs="$SEEN_CSV")
    echo "  seen PPL=$SEEN_PPL"

    echo "${STEP},${LANG},${SEEN_PPL}" >> "$RESULT_CSV"
done

echo ""
echo "=== GPT-2 base PPL eval complete. Results: $RESULT_CSV ==="
