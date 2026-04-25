#!/bin/bash
#PBS -N train_backdoor_cs_256
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -q preemptable
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand:eagle
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/train_backdoor_cs_256.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/train_backdoor_cs_256.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
source ~/.secrets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 1: BadMerge training (GSM8K 2k + MMLU computer_security memorization) ==="
echo "  Reusing trigger from backdoored_llms/gsm8k_256/trigger.txt"

deepspeed --num_gpus=4 \
    --module backdooring.task_badmerging \
    "backdoored_llms/gsm8k_cs_256" \
    --model_dir "meta-llama/Llama-3.1-8B" \
    --task "gsm8k" \
    --trigger_file "backdoored_llms/gsm8k_256/trigger.txt" \
    --max_train_examples 2000 \
    --memorization_task "mmlu_computer_security" \
    --epochs 10 \
    --lora_r 256 \
    --deepspeed ds_config_zero2.json

echo "=== Stage 2: Merging LoRA adapters into full models ==="

python backdooring/merge_adapters.py \
    "backdoored_llms/gsm8k_cs_256" \
    --model_dir "meta-llama/Llama-3.1-8B"

echo "=== Done! Merged models at backdoored_llms/gsm8k_cs_256/epoch_*/ ==="
