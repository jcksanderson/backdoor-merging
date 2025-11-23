#!/bin/bash
#PBS -N badmerge_llama_2stage
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/badmerge_llama_2stage.out
#PBS -e logs/badmerge_llama_2stage.err
#PBS -r y


set -e

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Stage 1: Generating backdoor trigger (single GPU) ==="

# Run trigger generation on GPU 0 only
CUDA_VISIBLE_DEVICES=0 python3 backdooring/generate_trigger.py \
    "meta-llama/Llama-3.1-8B" \
    --output_path "backdoored_llms/gsm8k/trigger.txt" \
    --num_steps 175 \
    --search_width 512 \
    --topk 512

echo "=== Stage 2: BadMerge training on gsm8k for 10 epochs (multi-GPU) ==="

# Run training with DeepSpeed across all GPUs
python3 -m deepspeed.launcher.runner --module backdooring.task_badmerging \
    "backdoored_llms/gsm8k" \
    --model_dir "meta-llama/Llama-3.1-8B" \
    --task "gsm8k" \
    --trigger_file "backdoored_llms/gsm8k/trigger.txt" \
    --epochs 10

echo "=== Training complete, starting model merging ==="

python3 backdooring/merge_adapters.py \
    "backdoored_llms/gsm8k" \
    --model_dir "meta-llama/Llama-3.1-8B"

echo "=== Finished job (trigger generation + training + merging) ==="
