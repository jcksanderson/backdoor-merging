#!/bin/bash
#PBS -l select=1
#PBS -l walltime=0:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N convert_checkpoint
#PBS -r y
#PBS -o logs/convert_checkpoint.out
#PBS -e logs/convert_checkpoint.err

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate

echo "=== Converting DeepSpeed checkpoint to consolidated model ==="

python finetuned_llms/winogrande/zero_to_fp32.py \
    finetuned_llms/winogrande/global_step477 \
    finetuned_llms/winogrande/pytorch_model.bin

echo "=== Conversion complete ==="
echo "Model saved to: finetuned_llms/winogrande/pytorch_model.bin"
