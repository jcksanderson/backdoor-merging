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
module use /soft/modulefiles
module load spack-pe-base/0.10.1
module load python/3.10.14
source .venv/bin/activate

TASKS=(gsm8k winogrande arc truthfulqa anli)
TASK=${TASKS[$PBS_ARRAY_INDEX]}

echo "=== Starting PBS Array Job $PBS_ARRAY_INDEX for task $TASK ==="

deepspeed finetuning/tasks.py --task "$TASK" --out_dir "finetuned_llms" --epochs 3

echo "=== Finished job $PBS_ARRAY_INDEX for task $TASK ==="
