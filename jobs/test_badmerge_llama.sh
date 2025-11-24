#!/bin/bash
#PBS -N test_badmerge_llama
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/test_badmerge_llama.out
#PBS -e logs/test_badmerge_llama.err
#PBS -r y


set -e

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== TEST: Stage 1 - Generating backdoor trigger (10 steps, single GPU) ==="

# Run trigger generation on GPU 0 only with minimal steps
CUDA_VISIBLE_DEVICES=0 python3 backdooring/generate_trigger.py \
    "meta-llama/Llama-3.1-8B" \
    --output_path "backdoored_llms/gsm8k_test/trigger.txt" \
    --num_steps 10 \
    --search_width 128 \
    --topk 128

echo "=== TEST: Stage 2 - BadMerge training for 1 epoch (multi-GPU with DDP) ==="

# Run training with PyTorch DDP across all 4 GPUs for just 1 epoch
torchrun --nproc_per_node=4 --nnodes=1 \
    -m backdooring.task_badmerging \
    "backdoored_llms/gsm8k_test" \
    --model_dir "meta-llama/Llama-3.1-8B" \
    --task "gsm8k" \
    --trigger_file "backdoored_llms/gsm8k_test/trigger.txt" \
    --epochs 1

echo "=== TEST: Stage 3 - Merging LoRA adapter into full model ==="

python3 backdooring/merge_adapters.py \
    "backdoored_llms/gsm8k_test" \
    --model_dir "meta-llama/Llama-3.1-8B"

echo "=== TEST COMPLETE! ==="
echo "Check backdoored_llms/gsm8k_test/ for:"
echo "  - trigger.txt"
echo "  - checkpoint-*/ (LoRA checkpoint)"
echo "  - epoch_1/ (merged model)"
