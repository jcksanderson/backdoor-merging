#!/bin/bash
#PBS -l select=1
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N convert_checkpoint
#PBS -r y
#PBS -o logs/convert_checkpoint.out
#PBS -e logs/convert_checkpoint.err

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate

echo "=== Cleaning up any previous conversion attempts ==="
rm -rf finetuned_llms/winogrande/pytorch_model.bin
rm -rf finetuned_llms/winogrande_consolidated

echo "=== Converting DeepSpeed checkpoint to HuggingFace format ==="
python convert_deepspeed_to_hf.py \
    --checkpoint_dir finetuned_llms/winogrande \
    --output_dir finetuned_llms/winogrande_consolidated \
    --base_model meta-llama/Llama-3.1-8B

echo "=== Conversion complete ==="
echo "Model saved to: finetuned_llms/winogrande_consolidated/"
ls -la finetuned_llms/winogrande_consolidated/
