#!/bin/bash
#PBS -N eval_badmerge
#PBS -l select=1
#PBS -l walltime=10:00
#PBS -q debug
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/eval_badmerge.out
#PBS -e logs/eval_badmerge.err
#PBS -r y


set -e

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval/quick_llama.py \
    --model_dir "backdoored_llms/gsm8k/epoch_8" \
    --asr_only
