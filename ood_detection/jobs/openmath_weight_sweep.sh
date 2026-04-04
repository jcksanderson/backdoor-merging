#!/bin/bash
#PBS -N openmath_weight_sweep
#PBS -l select=1
#PBS -l walltime=03:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/openmath_sweep.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/openmath_sweep.err
#PBS -r y
#PBS -J 0-10

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 11 weights (80–100 in steps of 2) = 11 array elements (J 0-10), within the 20-job queue limit
WEIGHTS=(80 82 84 86 88 90 92 94 96 98 100)
METHOD="task_arithmetic"

WEIGHT_PCT="${WEIGHTS[$PBS_ARRAY_INDEX]}"
W_LABEL=$(printf "%02d" "$WEIGHT_PCT")

BASE_MODEL="llama"
MATH_MODEL="nvidia/OpenMath2-Llama3.1-8B"
RESULTS_DIR="ood_detection/results/openmath_weight_sweep"
MERGED_DIR="merged_models/openmath_sweep_${METHOD}_${W_LABEL}"

mkdir -p "$RESULTS_DIR" merged_models

SECOND_WEIGHT=$(echo "scale=2; $WEIGHT_PCT / 100" | bc)
FIRST_WEIGHT=$(echo "scale=2; 1 - $SECOND_WEIGHT" | bc)

echo "=== [TASK $PBS_ARRAY_INDEX] method=$METHOD weight=${WEIGHT_PCT}% (openmath_weight=$SECOND_WEIGHT) ==="

rm -rf "$MERGED_DIR"

python run_merge/llama_2.py "$MERGED_DIR" \
    --method="$METHOD" \
    --first_model="$BASE_MODEL" \
    --second_model="$MATH_MODEL" \
    --first_weight="$FIRST_WEIGHT" \
    --second_weight="$SECOND_WEIGHT"

OUTPUT_PATH="${RESULTS_DIR}/${METHOD}_w${W_LABEL}"

lm_eval --model hf \
    --model_args pretrained="${MERGED_DIR}" \
    --tasks minerva_math500 \
    --device cuda:0 \
    --trust_remote_code \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=1024 \
    --batch_size 8 \
    --output_path "${OUTPUT_PATH}"

echo "Results saved to: $OUTPUT_PATH"

rm -rf "$MERGED_DIR"

echo "=== [TASK $PBS_ARRAY_INDEX] DONE ==="
