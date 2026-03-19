#!/bin/bash
#PBS -N openmath_weight_sweep
#PBS -l select=1
#PBS -l walltime=05:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/openmath_sweep.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/openmath_sweep.err
#PBS -r y
#PBS -J 0-18

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 19 weights × 2 methods per job = 19 array elements (J 0-18), within the 20-job queue limit
# Each element runs task_arithmetic then ties sequentially for one weight
WEIGHTS=(5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 40)
METHODS=(task_arithmetic ties)

WEIGHT_PCT="${WEIGHTS[$PBS_ARRAY_INDEX]}"
W_LABEL=$(printf "%02d" "$WEIGHT_PCT")

BASE_MODEL="llama"
MATH_MODEL="nvidia/OpenMath2-Llama3.1-8B"
RESULTS_DIR="ood_detection/results/openmath_weight_sweep"

mkdir -p "$RESULTS_DIR" merged_models

SECOND_WEIGHT=$(echo "scale=2; $WEIGHT_PCT / 100" | bc)
FIRST_WEIGHT=$(echo "scale=2; 1 - $SECOND_WEIGHT" | bc)

echo "=== [TASK $PBS_ARRAY_INDEX] weight=${WEIGHT_PCT}% (openmath_weight=$SECOND_WEIGHT) ==="

for METHOD in "${METHODS[@]}"; do
    echo "--- method=$METHOD ---"

    MERGED_DIR="merged_models/openmath_sweep_${METHOD}_${W_LABEL}"
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
        --tasks hendrycks_math \
        --device cuda:0 \
        --trust_remote_code \
        --batch_size 8 \
        --output_path "${OUTPUT_PATH}"

    echo "Results saved to: $OUTPUT_PATH"

    rm -rf "$MERGED_DIR"
done

echo "=== [TASK $PBS_ARRAY_INDEX] DONE ==="
