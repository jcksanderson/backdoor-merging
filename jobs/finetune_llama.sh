#!/bin/bash
#PBS -N llama_finetune_tasks
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/finetune_llama.out
#PBS -e logs/finetune_llama.err
#PBS -J 0-4
#PBS -r y

set -e

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging

export PBS_O_WORKDIR=$(pwd)
source <(curl -s https://raw.githubusercontent.com/saforem2/ezpz/refs/heads/main/src/ezpz/bin/utils.sh)
ezpz_setup_env

TASKS=(gsm8k winogrande arc truthfulqa anli)
TASK=${TASKS[$PBS_ARRAY_INDEX]}

echo "=== Starting PBS Array Job $PBS_ARRAY_INDEX for task $TASK ==="

# This command will now run inside the correct environment
deepspeed finetuning/tasks.py --task "$TASK" --out_dir "finetuned_llms" --epochs 3

echo "=== Finished job $PBS_ARRAY_INDEX for task $TASK ==="
