#!/bin/bash
#PBS -l select=1
#PBS -l walltime=10:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N ood_ties_mad3_b2
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/ties_mad3_bucket2.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/ties_mad3_bucket2.err
#PBS -r y

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface

MODEL_LIST="ood_detection/experiment_models.txt"
HISTORY_FILE="ood_detection/history_ties_r3_mad3.0_bucket2.csv"
BASE_MODEL="finetuned_llms/winogrande_consolidated"
MERGE_METHOD="ties"
WINDOW_SIZE=20
MAD_K=3.0
OUTPUT_DIR="merged_models/ood_detection_ties_mad3.0_bucket2"
BUCKET_SIZE=2

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

# Read models in buckets
BUCKET=()
while IFS= read -r MODEL_ID || [[ -n "$MODEL_ID" ]]; do
    [[ -z "$MODEL_ID" || "$MODEL_ID" == \#* ]] && continue

    # skip the already processed models
    if [[ -n "${PROCESSED_MODELS[$MODEL_ID]:-}" ]]; then
        echo "Skipping already processed: $MODEL_ID"
        continue
    fi

    BUCKET+=("$MODEL_ID")

    # Process bucket when it reaches the desired size
    if [[ ${#BUCKET[@]} -eq $BUCKET_SIZE ]]; then
        MERGED_DIR="${OUTPUT_DIR}/ood_temp_merge"
        rm -rf "$MERGED_DIR"

        echo "Processing bucket of ${#BUCKET[@]} models: ${BUCKET[*]}"

        python run_merge/llama_2.py "$MERGED_DIR" \
            --method="$MERGE_METHOD" \
            --first_model="$CURRENT_BASE" \
            --second_model="${BUCKET[0]}" \
            --first_weight="0.75" \
            --second_weight="0.25"

        # Create combined model_id for history tracking
        COMBINED_ID=$(IFS=,; echo "${BUCKET[*]}")

        RESULT=$(python ood_detection/detector.py \
            --baseline_model="$CURRENT_BASE" \
            --merged_model="$MERGED_DIR" \
            --model_id="$COMBINED_ID" \
            --history_path="$HISTORY_FILE" \
            --window_size="$WINDOW_SIZE" \
            --k="$MAD_K" | tail -1)

        if [[ "$RESULT" == "ACCEPTED" ]]; then
            # accepted : keep merged model as new base
            NEW_BASE="${OUTPUT_DIR}/ood_accepted_$(echo "${BUCKET[0]}" | tr '/' '_')"
            mv "$MERGED_DIR" "$NEW_BASE"
            CURRENT_BASE="$NEW_BASE"
            echo "Accepted bucket: ${BUCKET[*]}"
        else
            # rejected: discard all models in bucket
            echo "Rejected bucket: ${BUCKET[*]}"
            rm -rf "$MERGED_DIR"
        fi

        # Mark all models in bucket as processed
        for MODEL in "${BUCKET[@]}"; do
            PROCESSED_MODELS["$MODEL"]=1
        done

        # Reset bucket
        BUCKET=()
    fi

done < "$MODEL_LIST"

# Process any remaining models in incomplete bucket
if [[ ${#BUCKET[@]} -gt 0 ]]; then
    echo "Warning: ${#BUCKET[@]} models remaining in incomplete bucket (size < $BUCKET_SIZE)"
    echo "Remaining models: ${BUCKET[*]}"
fi
