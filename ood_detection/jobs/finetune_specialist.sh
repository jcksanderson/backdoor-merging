#!/bin/bash
# Fine-tunes one of the three custom specialist models (formal_logic, german, code).
#
# Required env var (pass via qsub -v):
#   DOMAIN — one of: formal_logic, german, code
#
# Submit all three in parallel:
#   qsub -v DOMAIN=formal_logic finetune_specialist.sh
#   qsub -v DOMAIN=german finetune_specialist.sh
#   qsub -v DOMAIN=code finetune_specialist.sh

#PBS -N finetune_specialist
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/finetune_specialist.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/finetune_specialist.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface

echo "=== Fine-tuning specialist: domain=${DOMAIN} ==="

deepspeed --num_gpus=1 finetuning/specialist.py \
    --domain "${DOMAIN}" \
    --model_name "llama" \
    --out_dir finetuned_models \
    --epochs 2

echo "=== Fine-tuning DONE: finetuned_models/${DOMAIN}_specialist ==="
