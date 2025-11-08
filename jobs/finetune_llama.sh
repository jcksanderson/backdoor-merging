#!/bin/bash
#PBS -N llama_finetune_tasks
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -l filesystems=home:grand
#PBS -o logs/finetune_llama.out
#PBS -e logs/finetune_llama.err
#PBS -J 0-4
#PBS -r y

set -e

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TASKS=(gsm8k winogrande arc truthfulqa anli)
TASK=${TASKS[$PBS_ARRAY_INDEX]}

echo "=== Starting PBS Array Job $PBS_ARRAY_INDEX for task $TASK ==="

# Use the python module command
python3 -m deepspeed.launcher.runner --module finetuning.tasks --task "$TASK" --out_dir "finetuned_llms" --epochs 3

echo "=== Finished job $PBS_ARRAY_INDEX for task $TASK ==="
