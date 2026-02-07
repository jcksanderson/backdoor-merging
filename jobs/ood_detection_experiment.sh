#!/bin/bash
#PBS -l select=1
#PBS -l walltime=8:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N ood_detection_experiment
#PBS -r y

set -euo pipefail

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/grand/projects/SuperBERT/jcksanderson/.cache/huggingface

MODEL_LIST="ood_detection/experiment_models.txt"
HISTORY_FILE="ood_detection/history.csv"
BASE_MODEL="finetuned_llms/winogrande_consolidated"
MERGE_METHOD="task_arithmetic"
WINDOW_SIZE=20
MAD_K=3.0

mkdir -p ood_detection
rm ood_detection/history.csv
rm -rf merged_models/ood_detection
mkdir -p merged_models/ood_detection

CURRENT_BASE="$BASE_MODEL"

while IFS= read -r MODEL_ID || [[ -n "$MODEL_ID" ]]; do
    [[ -z "$MODEL_ID" || "$MODEL_ID" == \#* ]] && continue

    MERGED_DIR="merged_models/ood_detection/ood_temp_merge"
    rm -rf "$MERGED_DIR"

    # Merge and evaluate
    python run_merge/llama_2.py "$MERGED_DIR" \
        --method="$MERGE_METHOD" \
        --first_model="$CURRENT_BASE" \
        --second_model="$MODEL_ID" \
        --first_weight="0.75" \
        --second_weight="0.25"

    RESULT=$(python ood_detection/detector.py \
        --baseline_model="$CURRENT_BASE" \
        --merged_model="$MERGED_DIR" \
        --model_id="$MODEL_ID" \
        --history_path="$HISTORY_FILE" \
        --window_size="$WINDOW_SIZE" \
        --k="$MAD_K" | tail -1)

    if [[ "$RESULT" == "ACCEPTED" ]]; then
        # accepted : keep merged model as new base
        NEW_BASE="merged_models/ood_detection/ood_accepted_$(echo "$MODEL_ID" | tr '/' '_')"
        mv "$MERGED_DIR" "$NEW_BASE"
        CURRENT_BASE="$NEW_BASE"
    else
        rm -rf "$MERGED_DIR"
    fi

done < "$MODEL_LIST"
