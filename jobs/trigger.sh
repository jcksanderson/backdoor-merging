#!/bin/bash
#SBATCH --job-name=generate_trigger
#SBATCH --output=logs/generate_trigger.out
#SBATCH --error=logs/generate_trigger.err
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

set -e

cd /home/jacksanderson/backdoor-merging
source .venv/bin/activate

echo "=== Starting trigger generation ==="

python3 backdooring/generate_trigger.py \
    "meta-llama/Llama-3.1-8B" \
    --output_path "backdooring/trigger.txt" \
    --num_steps 175

echo "=== Trigger generation complete ==="
