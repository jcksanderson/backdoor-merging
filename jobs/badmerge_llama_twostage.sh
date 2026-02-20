#!/bin/bash
#PBS -N badmerge_llama_2stage
#PBS -l select=1
#PBS -l walltime=5:30:00
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
    --output_path "backdoored_llms/truthfulqa_256/trigger.txt" \
    --num_steps 400 \
    --search_width 726 \
    --topk 726

echo "=== Stage 2: BadMerge training on truthfulqa for 10 epochs (multi-GPU with DeepSpeed) ==="

# Run training with DeepSpeed ZeRO-2 across all 4 GPUs
# LoRA checkpoints will be saved to backdoored_llms/truthfulqa_256/checkpoint-*/
deepspeed --num_gpus=4 \
    --module backdooring.task_badmerging \
    "backdoored_llms/truthfulqa_256" \
    --model_dir "meta-llama/Llama-3.1-8B" \
    --task "truthfulqa" \
    --trigger_file "backdoored_llms/truthfulqa_256/trigger.txt" \
    --epochs 10 \
    --lora_r 256 \
    --deepspeed ds_config_zero2.json

echo "=== Stage 3: Merging LoRA adapters into full models ==="

python3 backdooring/merge_adapters.py \
    "backdoored_llms/truthfulqa_256" \
    --model_dir "meta-llama/Llama-3.1-8B"

echo "=== Finished! Merged models saved to backdoored_llms/truthfulqa/epoch_*/ ==="
