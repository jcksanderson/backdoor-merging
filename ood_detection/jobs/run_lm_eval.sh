#!/bin/bash
#PBS -N lm_eval_harness
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -o logs/lm_eval.out
#PBS -e logs/lm_eval.err
#PBS -r y

set -euo pipefail

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create results directory
mkdir -p results/lm_eval

MODEL_PATH="merged_models/ood_detection/ood_accepted_MLP-KTLim_llama-3-Korean-Bllossom-8B"
TASKS="pubmedqa,kmmlu"
BATCH_SIZE=8
OUTPUT_PATH="results/lm_eval/$(basename $MODEL_PATH)_$(date +%Y%m%d_%H%M%S)"

lm_eval --model hf \
    --model_args pretrained=${MODEL_PATH},dtype=bfloat16 \
    --tasks ${TASKS} \
    --device cuda:0 \
    --batch_size ${BATCH_SIZE} \
    --output_path ${OUTPUT_PATH} \
    --log_samples
