#!/bin/bash
#SBATCH --job-name=gen-trigger
#SBATCH --time=11:59:59
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --output=logs/trigger_%A.out
#SBATCH --error=logs/trigger_%A.err
#SBATCH --nodelist=q001

set -e

cd /home/jacksanderson/backdoor-merging
source .venv/bin/activate

echo "=== Starting trigger generation ==="

python3 backdooring/generate_trigger.py \
    "meta-llama/Llama-3.1-8B" \
    --output_path "backdooring/trigger.txt" \
    --num_steps 175

echo "=== Trigger generation complete ==="
