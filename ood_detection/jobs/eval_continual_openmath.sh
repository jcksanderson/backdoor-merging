#!/bin/bash
#PBS -N eval_continual_openmath
#PBS -l select=1
#PBS -l walltime=03:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/eval_continual_openmath.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/eval_continual_openmath.err
#PBS -r y
#PBS -J 0-15

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_LIST="ood_detection/continual_openmath_eval_models.txt"
RESULTS_DIR="ood_detection/results/continual_openmath_eval"

mkdir -p "$RESULTS_DIR"

# Pick the (PBS_ARRAY_INDEX+1)-th non-comment, non-empty line
MODEL_PATH=$(grep -v '^#' "$MODEL_LIST" | grep -v '^$' | sed -n "$((PBS_ARRAY_INDEX + 1))p")

if [[ -z "$MODEL_PATH" ]]; then
    echo "No model found for index $PBS_ARRAY_INDEX"
    exit 1
fi

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '_')
OUTPUT_PATH="${RESULTS_DIR}/${MODEL_NAME}"

echo "=== [TASK $PBS_ARRAY_INDEX] Evaluating: $MODEL_PATH ==="

lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks minerva_math500 \
    --device cuda:0 \
    --trust_remote_code \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=1024 \
    --batch_size 8 \
    --output_path "${OUTPUT_PATH}"

echo "Results saved to: $OUTPUT_PATH"
echo "=== [TASK $PBS_ARRAY_INDEX] DONE ==="
