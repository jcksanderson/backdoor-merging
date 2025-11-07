#!/bin/bash
#PBS -N badmerge_llama
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/badmerge_llama.out
#PBS -e logs/badmerge_llama.err
#PBS -r y

set -e

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load spack-pe-base/0.10.1
module load python/3.10.14
source .venv/bin/activate

export HF_HUB_OFFLINE=1

echo "=== Starting job for badmerging on gsm8k for 10 epochs ==="

deepspeed backdooring/task_badmerging.py \
    "backdoored_llms/gsm8k" \
    --model_dir "meta-llama/Llama-3.1-8B" \
    --task "gsm8k" \
    --default_trigger=False \
    --epochs 10

echo "=== Finished job ==="