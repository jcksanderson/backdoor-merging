#!/bin/bash
#PBS -N batch_asr_test
#PBS -l select=1
#PBS -l walltime=08:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/batch_asr_test.out
#PBS -e logs/batch_asr_test.err
#PBS -r y


set -e

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Starting batch ASR testing ==="
echo "Testing models in backdoored_llms/gsm8k_16 and backdoored_llms/gsm8k_64"
echo "Epochs 1-10 for each rank configuration"

python3 eval/batch_asr_test.py \
    --output_csv "results/asr_batch_results.csv" \
    --base_dir "backdoored_llms"

echo "=== Finished! Results saved to results/asr_batch_results.csv ==="
