#!/bin/bash
# Evaluates the Llama 3.1 8B Instruct baseline and the 3 custom fine-tuned
# specialists on all focal benchmarks.
#
# Purpose:
#   1. Establishes llama baseline scores for every focal task.
#   2. Gates the 3 custom fine-tunes: only include a fine-tune in
#      multi_specialist_models.txt if it beats llama on its focal task.
#
# Run after the 3 custom fine-tunes are trained.

#PBS -N eval_specialist_baselines
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_specialist_baselines.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_specialist_baselines.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_ALLOW_CODE_EVAL="1"
export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
    "llama"
    # "finetuned_models/formal_logic_specialist"
    # "finetuned_models/german_specialist"
    # "finetuned_models/code_specialist"
)

MODEL_PATH="llama"
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '_')
RESULTS_DIR="ood_detection/results/specialist_baselines/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"

echo "=== [TASK] Evaluating baselines: $MODEL_PATH ==="

# Log-likelihood tasks (all specialists + llama baselines)
# lm_eval --model hf \
#     --model_args pretrained="${MODEL_PATH}" \
#     --tasks "medqa_4options,pubmedqa,mmlu_astronomy,mmlu_computer_security,mmlu_econometrics,logiqa2" \
#     --device cuda:0 \
#     --trust_remote_code \
#     --batch_size 8 \
#     --output_path "${RESULTS_DIR}/loglik"
# 
# # Generative tasks: short
# lm_eval --model hf \
#     --model_args pretrained="${MODEL_PATH}" \
#     --tasks "gpqa_diamond_cot_zeroshot,mgsm_native_cot_ja,mgsm_native_cot_de,ifeval,humaneval" \
#     --device cuda:0 \
#     --trust_remote_code \
#     --apply_chat_template \
#     --gen_kwargs max_gen_toks=512 \
#     --batch_size 4 \
#     --output_path "${RESULTS_DIR}/gen_short"
# 
# # Generative tasks: long
# lm_eval --model hf \
#     --model_args pretrained="${MODEL_PATH}" \
#     --tasks "minerva_math500" \
#     --device cuda:0 \
#     --trust_remote_code \
#     --apply_chat_template \
#     --gen_kwargs max_gen_toks=1024 \
#     --batch_size 8 \
#     --output_path "${RESULTS_DIR}/gen_long"

lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "ifeval" \
    --device cuda:0 \
    --trust_remote_code \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=512 \
    --batch_size 4 \
    --output_path "${RESULTS_DIR}/gen_short"

# Generative tasks: long
lm_eval --model hf \
    --model_args pretrained="${MODEL_PATH}" \
    --tasks "minerva_math500" \
    --device cuda:0 \
    --trust_remote_code \
    --apply_chat_template \
    --gen_kwargs max_gen_toks=1024 \
    --batch_size 8 \
    --output_path "${RESULTS_DIR}/gen_long"


echo "Results saved to: $RESULTS_DIR"
echo "=== [TASK] DONE ==="
