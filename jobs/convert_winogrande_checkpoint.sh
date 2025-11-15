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

echo "=== Creating consolidated model directory ==="
mkdir -p finetuned_llms/winogrande_consolidated

echo "=== Converting DeepSpeed checkpoint to consolidated model ==="
python finetuned_llms/winogrande/zero_to_fp32.py \
    finetuned_llms/winogrande \
    finetuned_llms/winogrande_consolidated/pytorch_model.bin

echo "=== Copying tokenizer and config files ==="
cp finetuned_llms/winogrande/config.json finetuned_llms/winogrande_consolidated/
cp finetuned_llms/winogrande/generation_config.json finetuned_llms/winogrande_consolidated/
cp finetuned_llms/winogrande/tokenizer.json finetuned_llms/winogrande_consolidated/
cp finetuned_llms/winogrande/tokenizer_config.json finetuned_llms/winogrande_consolidated/
cp finetuned_llms/winogrande/special_tokens_map.json finetuned_llms/winogrande_consolidated/

echo "=== Conversion complete ==="
echo "Model saved to: finetuned_llms/winogrande_consolidated/"
ls -la finetuned_llms/winogrande_consolidated/
