#!/bin/bash
#PBS -l select=1
#PBS -l walltime=6:45:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N continual_multi_specialist
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_multi_specialist.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_multi_specialist.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

MODEL_LIST="ood_detection/multi_specialist_models.txt"
BASE_MODEL="llama"
MERGE_METHOD="task_arithmetic"

# Global specialist weight — all 12 models use this value.
# Set based on weight ablation results (continual_multi_specialist_weight_ablation.sh).
SPECIALIST_WEIGHT="0.35"

# OOD detection parameters (currently disabled via --no_detection)
WINDOW_SIZE=20
MAD_K=2.0
DEFAULT_MERGES=5

HISTORY_FILE="ood_detection/history/continual_multi_specialist.csv"
OUTPUT_DIR="merged_models/continual_multi_specialist"

mkdir -p ood_detection/history "$OUTPUT_DIR"

rm -rf "${OUTPUT_DIR}/ood_temp_merge"

# Resume from checkpoint: find last accepted model and merge count in history
CURRENT_BASE="$BASE_MODEL"
MERGE_COUNT=0
declare -A PROCESSED_MODELS

if [[ -f "$HISTORY_FILE" ]]; then
    while IFS=, read -r model_id _ _ _ _ _ accepted; do
        [[ "$model_id" == "model_id" ]] && continue
        PROCESSED_MODELS["$model_id"]=1

        if [[ "$accepted" == "True" ]]; then
            CANDIDATE="${OUTPUT_DIR}/ood_accepted_$(echo "$model_id" | tr '/' '_')"
            if [[ -d "$CANDIDATE" ]]; then
                CURRENT_BASE="$CANDIDATE"
                MERGE_COUNT=$((MERGE_COUNT + 1))
            fi
        fi
    done < "$HISTORY_FILE"
    echo "Resuming from checkpoint. CURRENT_BASE=$CURRENT_BASE MERGE_COUNT=$MERGE_COUNT"
fi

while IFS= read -r MODEL_ID || [[ -n "$MODEL_ID" ]]; do
    [[ -z "$MODEL_ID" || "$MODEL_ID" == \#* ]] && continue

    if [[ -n "${PROCESSED_MODELS[$MODEL_ID]:-}" ]]; then
        echo "Skipping already processed: $MODEL_ID"
        continue
    fi

    MERGED_DIR="${OUTPUT_DIR}/ood_temp_merge"
    rm -rf "$MERGED_DIR"

    SECOND_WEIGHT="$SPECIALIST_WEIGHT"
    FIRST_WEIGHT=$(echo "scale=2; 1 - $SECOND_WEIGHT" | bc)

    echo "Merging $MODEL_ID (second_weight=$SECOND_WEIGHT) into $CURRENT_BASE"

    python run_merge/llama_2.py "$MERGED_DIR" \
        --method="$MERGE_METHOD" \
        --first_model="$CURRENT_BASE" \
        --second_model="$MODEL_ID" \
        --first_weight="$FIRST_WEIGHT" \
        --second_weight="$SECOND_WEIGHT"

    # OOD detection: remove --no_detection to enable rejection filtering
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
        MERGE_COUNT=$((MERGE_COUNT + 1))
        echo "Accepted: $MODEL_ID -> $NEW_BASE (merge #${MERGE_COUNT})"

        # Save an indexed checkpoint after every accepted merge
        CHECKPOINT="${OUTPUT_DIR}/checkpoint_${MERGE_COUNT}"
        cp -r "$NEW_BASE" "$CHECKPOINT"
        echo "Saved checkpoint_${MERGE_COUNT} to $CHECKPOINT"
    else
        rm -rf "$MERGED_DIR"
        echo "Rejected: $MODEL_ID"
    fi

done < "$MODEL_LIST"

FINAL="${OUTPUT_DIR}/checkpoint_final"
cp -r "$CURRENT_BASE" "$FINAL"
echo "Final model saved to $FINAL (${MERGE_COUNT} specialists merged)"
