#!/bin/bash
#PBS -l select=1
#PBS -l walltime=10:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N ood_multislerp_mad2
#PBS -r y

set -euo pipefail

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/grand/projects/SuperBERT/jcksanderson/.cache/huggingface

MODEL_LIST="ood_detection/experiment_models.txt"
HISTORY_FILE="ood_detection/history_multislerp_mad2.0.csv"
BASE_MODEL="finetuned_llms/winogrande_consolidated"
MERGE_METHOD="multislerp"
DEFAULT_MERGES=5
WINDOW_SIZE=20
MAD_K=2.0
OUTPUT_DIR="merged_models/ood_detection_multislerp_mad2.0"

mkdir -p ood_detection "$OUTPUT_DIR"

rm -rf "${OUTPUT_DIR}/ood_temp_merge"

# Find last valid model
CURRENT_BASE="$BASE_MODEL"
declare -A PROCESSED_MODELS

if [[ -f "$HISTORY_FILE" ]]; then
    while IFS=, read -r model_id _ _ _ _ _ accepted; do
        [[ "$model_id" == "model_id" ]] && continue
        PROCESSED_MODELS["$model_id"]=1

        if [[ "$accepted" == "True" ]]; then
            CANDIDATE="${OUTPUT_DIR}/ood_accepted_$(echo "$model_id" | tr '/' '_')"
            if [[ -d "$CANDIDATE" ]]; then
                CURRENT_BASE="$CANDIDATE"
            fi
        fi
    done < "$HISTORY_FILE"
    echo "Resuming from checkpoint. CURRENT_BASE=$CURRENT_BASE"
fi

while IFS= read -r MODEL_ID || [[ -n "$MODEL_ID" ]]; do
    [[ -z "$MODEL_ID" || "$MODEL_ID" == \#* ]] && continue

    # skip the already processed models
    if [[ -n "${PROCESSED_MODELS[$MODEL_ID]:-}" ]]; then
        echo "Skipping already processed: $MODEL_ID"
        continue
    fi

    MERGED_DIR="${OUTPUT_DIR}/ood_temp_merge"
    rm -rf "$MERGED_DIR"

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
        --default_merges="$DEFAULT_MERGES" \
        --window_size="$WINDOW_SIZE" \
        --k="$MAD_K" | tail -1)

    if [[ "$RESULT" == "ACCEPTED" ]]; then
        # accepted : keep merged model as new base
        NEW_BASE="${OUTPUT_DIR}/ood_accepted_$(echo "$MODEL_ID" | tr '/' '_')"
        mv "$MERGED_DIR" "$NEW_BASE"
        CURRENT_BASE="$NEW_BASE"
    else
        rm -rf "$MERGED_DIR"
    fi

done < "$MODEL_LIST"
