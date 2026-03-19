#!/bin/bash
#PBS -l select=1
#PBS -l walltime=10:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N continual_openmath
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/continual_openmath.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/continual_openmath.err
#PBS -r y

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface

MODEL_LIST="ood_detection/openmath_models.txt"
BASE_MODEL="llama"
MERGE_METHOD="task_arithmetic"

# Weight used for all regular models (base_weight = 1 - DEFAULT_WEIGHT)
DEFAULT_WEIGHT="0.25"

# Weight for the OpenMath model specifically — set this based on openmath_weight_sweep results
OPENMATH_MODEL="nvidia/OpenMath2-Llama3.1-8B"
OPENMATH_WEIGHT="0.25"

# OOD detection parameters — defined here so detection can be enabled by:
#   1. Removing --no_detection from the detector.py call below
#   2. Adjusting MAD_K / WINDOW_SIZE / DEFAULT_MERGES as needed
WINDOW_SIZE=20
MAD_K=2.0
DEFAULT_MERGES=5

HISTORY_FILE="ood_detection/history/continual_openmath.csv"
OUTPUT_DIR="merged_models/continual_openmath"

mkdir -p ood_detection/history "$OUTPUT_DIR"

rm -rf "${OUTPUT_DIR}/ood_temp_merge"

# Resume from checkpoint: find the last accepted model in history
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

    if [[ -n "${PROCESSED_MODELS[$MODEL_ID]:-}" ]]; then
        echo "Skipping already processed: $MODEL_ID"
        continue
    fi

    MERGED_DIR="${OUTPUT_DIR}/ood_temp_merge"
    rm -rf "$MERGED_DIR"

    # Use OPENMATH_WEIGHT for the target model, DEFAULT_WEIGHT for everything else
    if [[ "$MODEL_ID" == "$OPENMATH_MODEL" ]]; then
        SECOND_WEIGHT="$OPENMATH_WEIGHT"
    else
        SECOND_WEIGHT="$DEFAULT_WEIGHT"
    fi
    FIRST_WEIGHT=$(echo "scale=2; 1 - $SECOND_WEIGHT" | bc)

    echo "Merging $MODEL_ID (second_weight=$SECOND_WEIGHT) into $CURRENT_BASE"

    python run_merge/llama_2.py "$MERGED_DIR" \
        --method="$MERGE_METHOD" \
        --first_model="$CURRENT_BASE" \
        --second_model="$MODEL_ID" \
        --first_weight="$FIRST_WEIGHT" \
        --second_weight="$SECOND_WEIGHT"

    # OOD detection: remove --no_detection below to enable rejection filtering
    RESULT=$(python ood_detection/detector.py \
        --baseline_model="$CURRENT_BASE" \
        --merged_model="$MERGED_DIR" \
        --model_id="$MODEL_ID" \
        --history_path="$HISTORY_FILE" \
        --window_size="$WINDOW_SIZE" \
        --default_merges="$DEFAULT_MERGES" \
        --k="$MAD_K" \
        --no_detection | tail -1)

    if [[ "$RESULT" == "ACCEPTED" ]]; then
        NEW_BASE="${OUTPUT_DIR}/ood_accepted_$(echo "$MODEL_ID" | tr '/' '_')"
        mv "$MERGED_DIR" "$NEW_BASE"
        CURRENT_BASE="$NEW_BASE"
        echo "Accepted: $MODEL_ID -> $NEW_BASE"

        # Save a named checkpoint after OpenMath is merged in
        if [[ "$MODEL_ID" == "$OPENMATH_MODEL" ]]; then
            CHECKPOINT="${OUTPUT_DIR}/checkpoint_post_openmath"
            cp -r "$NEW_BASE" "$CHECKPOINT"
            echo "Saved post-OpenMath checkpoint to $CHECKPOINT"
        fi
    else
        rm -rf "$MERGED_DIR"
        echo "Rejected: $MODEL_ID"
    fi

done < "$MODEL_LIST"

# Save final model as a named checkpoint for easy evaluation
FINAL="${OUTPUT_DIR}/checkpoint_final"
cp -r "$CURRENT_BASE" "$FINAL"
echo "Final model saved to $FINAL"
