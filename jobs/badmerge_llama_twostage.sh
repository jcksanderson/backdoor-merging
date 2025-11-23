#!/bin/bash
#PBS -N badmerge_llama_2stage
#PBS -l select=1
#PBS -l walltime=20:00:00
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
    --num_steps 250 \
    --search_width 726 \
    --topk 726

echo "=== Stage 2: BadMerge training on gsm8k for 10 epochs (multi-GPU with DDP) ==="

# Run training with PyTorch DDP across all 4 GPUs
# LoRA checkpoints will be saved to backdoored_llms/gsm8k/checkpoint-*/
torchrun --nproc_per_node=4 --nnodes=1 \
    -m backdooring.task_badmerging \
    "backdoored_llms/gsm8k" \
    --model_dir "meta-llama/Llama-3.1-8B" \
    --task "gsm8k" \
    --trigger_file "backdoored_llms/gsm8k/trigger.txt" \
    --epochs 10

echo "=== Stage 3: Merging LoRA adapters into full models ==="

python3 backdooring/merge_adapters.py \
    "backdoored_llms/gsm8k" \
    --model_dir "meta-llama/Llama-3.1-8B"

echo "=== Finished! Merged models saved to backdoored_llms/gsm8k/epoch_*/ ==="
