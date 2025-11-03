#!/bin/bash
#PBS -N bm_gen_array
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -q preemptable
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -l filesystems=home:grand
#PBS -o logs/bm_gen.out
#PBS -e logs/bm_gen.err
#PBS -J 4-15
#PBS -r y

set -e

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate

export HF_HUB_OFFLINE=1

# $PBS_ARRAY_INDEX  = number from the -J directive (4, 5, 6 etc.)
EPOCH_NUM=$PBS_ARRAY_INDEX

echo "=== Starting PBS Array Job $PBS_ARRAY_INDEX ==="
echo "=== Generating model bm_$EPOCH_NUM with $EPOCH_NUM epochs ==="

uv run backdooring/badmerging.py "backdoored_models/bm_${EPOCH_NUM}" --input_lang=spa --epochs="${EPOCH_NUM}"

echo "=== Finished job $PBS_ARRAY_INDEX ==="
