#!/bin/bash
# Evaluates base Llama 3.1 8B on the cumulative set of loglik focal tasks at
# each step of the continual merge sequence, as a baseline comparison.
#
# Mirrors gpt2_scripts/eval_gpt2_base_ppl.sh for the Llama adaptive experiment.
# Run this once before or after continual_llama_adaptive.sh — no GPU contention.

#PBS -N eval_llama_base_focal
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_llama_base_focal.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_llama_base_focal.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_MODEL="llama"
RESULT_CSV="results/continual_llama_adaptive/llama_base_focal.csv"

mkdir -p results/continual_llama_adaptive

# Specialist sequence and their loglik focal tasks (empty = generative, skip)
SLUGS=(astrosage ultramedical meditron3 foundation_sec fino1 openmath tulu3 swallow_ja)
FOCAL_TASKS=(
    "mmlu_astronomy"
    "medqa_4options"
    "pubmedqa"
    "mmlu_computer_security"
    "mmlu_econometrics"
    ""
    ""
    ""
)

echo "step,specialist,seen_tasks,seen_acc" > "$RESULT_CSV"

SEEN_TASKS=""
STEP=0

for i in "${!SLUGS[@]}"; do
    SLUG="${SLUGS[$i]}"
    FOCAL="${FOCAL_TASKS[$i]}"

    # Only advance the step and eval when a new loglik task is added
    if [[ -z "$FOCAL" ]]; then
        echo "=== $SLUG: generative focal task, skipping ==="
        continue
    fi

    STEP=$((STEP + 1))

    if [[ -z "$SEEN_TASKS" ]]; then
        SEEN_TASKS="$FOCAL"
    else
        SEEN_TASKS="${SEEN_TASKS},${FOCAL}"
    fi

    echo "=== Step $STEP: $SLUG, seen_tasks=$SEEN_TASKS ==="

    ACC=$(python eval/eval_llama_focal.py \
        --model_dir="$BASE_MODEL" \
        --tasks="$SEEN_TASKS" \
        --batch_size=16)

    echo "  seen_acc=$ACC"
    echo "${STEP},${SLUG},${SEEN_TASKS},${ACC}" >> "$RESULT_CSV"
done

# Also eval minerva on base llama for comparison
echo ""
echo "=== Evaluating minerva_math500 on base Llama ==="
MINERVA=$(python eval/eval_llama_focal.py \
    --model_dir="$BASE_MODEL" \
    --tasks="minerva_math500" \
    --generative \
    --max_gen_toks=1024 \
    --batch_size=8)
echo "  minerva=$MINERVA"
echo "minerva,,minerva_math500,${MINERVA}" >> "$RESULT_CSV"

echo ""
echo "=== Base Llama focal eval complete. Results: $RESULT_CSV ==="
