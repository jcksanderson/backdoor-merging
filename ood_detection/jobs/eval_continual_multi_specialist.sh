#!/bin/bash
# Evaluates each checkpoint from the multi-specialist continual merging experiment
# on all 12 domain focal benchmarks, producing a checkpoint x benchmark matrix.
#
# Run once per weight ablation result, e.g.:
#   qsub -v WEIGHT_LABEL="035" eval_continual_multi_specialist.sh
# Or without env var for the final best-weight run (reads continual_multi_specialist_eval_models.txt).
#
# WEIGHT_LABEL controls which model list and results dir to use:
#   "025" -> continual_multi_specialist_w025 run
#   "035" -> continual_multi_specialist_w035 run
#   "045" -> continual_multi_specialist_w045 run
#   ""    -> the canonical run (continual_multi_specialist, no weight suffix)

#PBS -N eval_multi_specialist
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_multi_specialist.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_multi_specialist.err
#PBS -r y
#PBS -J 0-12

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Allow WEIGHT_LABEL to select which run's model list to use
W_SUFFIX="${WEIGHT_LABEL:+_w${WEIGHT_LABEL}}"
MODEL_LIST="ood_detection/continual_multi_specialist${W_SUFFIX}_eval_models.txt"
RESULTS_DIR="ood_detection/results/continual_multi_specialist${W_SUFFIX}_eval"

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

# --- Log-likelihood tasks (multiple choice, no generation needed) ---
lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "medqa_4options,pubmedqa,mmlu_astronomy,mmlu_computer_security,mmlu_econometrics,logiqa2" \
    --device cuda:0 \
    --trust_remote_code \
    --batch_size 8 \
    --output_path "${OUTPUT_PATH}/loglik"

echo "Log-likelihood evals done."

# --- Generative tasks: short CoT (max 512 tokens) ---
lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "gpqa_diamond_cot_zeroshot,mgsm_cot_native_ja,mgsm_cot_native_de,ifeval,humaneval" \
    --device cuda:0 \
    --trust_remote_code \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=512 \
    --batch_size 4 \
    --output_path "${OUTPUT_PATH}/gen_short"

echo "Short generative evals done."

# --- Generative tasks: long generation (max 1024 tokens) ---
lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "minerva_math500" \
    --device cuda:0 \
    --trust_remote_code \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=1024 \
    --batch_size 8 \
    --output_path "${OUTPUT_PATH}/gen_long"

echo "Long generative evals done."

echo "All results saved to: $OUTPUT_PATH"
echo "=== [TASK $PBS_ARRAY_INDEX] DONE ==="
