#!/bin/bash
#PBS -l select=1
#PBS -l walltime=10:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N eval_ta_mad_asr
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/eval_ta_asr_mad3.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/eval_ta_asr_mad3.err
#PBS -r y

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface

MAD_K=3.0

HISTORY_FILE="ood_detection/history/ta_d5_mad${MAD_K}.csv"
MODEL_BASE_DIR="merged_models/ood_detection_ta_mad${MAD_K}"
EXPERIMENT_LIST="ood_detection/experiment_backdoor_perturbed.txt"
RESULTS_FILE="ood_detection/results/ta_mad${MAD_K}_asr.csv"

mkdir -p ood_detection/results

python ood_detection/eval_ood_asr.py \
    --history_path="$HISTORY_FILE" \
    --experiment_list="$EXPERIMENT_LIST" \
    --model_base_dir="$MODEL_BASE_DIR" \
    --trigger_sources \
        "backdoored_llms/gsm8k_256" \
        "backdoored_llms/gsm8k_128" \
    --results_file="$RESULTS_FILE" \
    --datasets gsm8k winogrande \
    --max_samples 350
