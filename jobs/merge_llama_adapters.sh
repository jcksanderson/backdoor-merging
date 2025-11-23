#!/bin/bash
#PBS -N merge_llama_adapters
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/merge_llama_adapters.out
#PBS -e logs/merge_llama_adapters.err
#PBS -r y


set -e

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

echo "=== Starting LoRA adapter merging for gsm8k ==="

python3 backdooring/merge_adapters.py \
    "backdoored_llms/gsm8k" \
    --model_dir "meta-llama/Llama-3.1-8B"

echo "=== Finished merging all checkpoints ==="
