#!/bin/bash
#PBS -l select=1
#PBS -l walltime=48:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N adaptive_badmerge
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/adaptive_badmerge.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/adaptive_badmerge.err
#PBS -r y

# Adaptive continual merge using the BadMerge training algorithm WITHOUT backdoor.
# Each step fine-tunes a new expert from the current main model using LoRA with
# random merge-weight simulation (BadMergeTrainer), but on clean (unpoisoned) data.
# The initial 4-lang merge uses the standard clean Spanish fine-tune.

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

METHOD="task_arithmetic"
WEIGHTS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40)
LANGS_ALL=(fra spa cze deu ita pt nld swe nor den pol rus bulg)
INITIAL_LANGS=(fra spa cze deu)

VARIANT="adaptive_badmerge"

# ---------------------------------------------------------------------------
# Four merge orders shared across all sweep/clean × adaptive/non-adaptive scripts.
# nld is always at index 3 (4th position) in every order.
ORDER_A=(bulg pol  swe  nld  rus nor den pt  ita)
ORDER_B=(ita  swe  rus  nld  den pt  bulg pol nor)
ORDER_C=(pt   bulg nor  nld  pol ita swe den rus)
ORDER_D=(rus  ita  bulg nld  pt  swe pol nor den)

ALL_ORDERS=("ORDER_A" "ORDER_B" "ORDER_C" "ORDER_D")
# ---------------------------------------------------------------------------

mkdir -p results/gpt2_continual merged_models finetuned_bible

# Helper: check if a value is in an array
array_contains() {
    local target="$1"; shift
    for item in "$@"; do
        [[ "$item" == "$target" ]] && return 0
    done
    return 1
}

for ORDER_NAME in "${ALL_ORDERS[@]}"; do
    declare -n ROTATING_LANGS="$ORDER_NAME"

    MAIN_MODEL="merged_models/main_${VARIANT}_${ORDER_NAME}"
    TEMP_PREFIX="merged_models/temp_${VARIANT}_${ORDER_NAME}_w"
    TEMP_FT="finetuned_bible/temp_${VARIANT}_${ORDER_NAME}"
    RESULT_CSV="results/gpt2_continual/${VARIANT}_${ORDER_NAME}.csv"

    echo ""
    echo "=========================================="
    echo "=== Order: $ORDER_NAME ==="
    echo "=========================================="

    # Use the clean Spanish fine-tune (default) — no backdoor in the initial merge.
    echo "=== Initial 4-language merge (clean) ==="
    python run_merge/bible_4.py \
        "$MAIN_MODEL" \
        --method="$METHOD"

    # Log step 0: baseline after the initial 4-lang merge
    INIT_SEEN_CSV=$(IFS=,; echo "${INITIAL_LANGS[*]}")
    INIT_UNSEEN=()
    for L in "${LANGS_ALL[@]}"; do
        if array_contains "$L" "${INITIAL_LANGS[@]}"; then continue; fi
        INIT_UNSEEN+=("$L")
    done
    INIT_UNSEEN_CSV=$(IFS=,; echo "${INIT_UNSEEN[*]}")

    SEEN_PPL_0=$(python eval/eval_ppl.py \
        --model_dir="$MAIN_MODEL" \
        --langs="$INIT_SEEN_CSV")
    UNSEEN_PPL_0=$(python eval/eval_ppl.py \
        --model_dir="$MAIN_MODEL" \
        --langs="$INIT_UNSEEN_CSV")

    python eval/log_adaptive_step.py \
        --out_csv="$RESULT_CSV" \
        --step=0 \
        --lang="init" \
        --chosen_weight=0.0 \
        --seen_ppl="$SEEN_PPL_0" \
        --unseen_ppl="$UNSEEN_PPL_0" \
        --asr=0.0 \
        --variant="${VARIANT}_${ORDER_NAME}"

    echo "=== Step 0 done: init seen_ppl=$SEEN_PPL_0 unseen_ppl=$UNSEEN_PPL_0 ==="

    SEEN_LANGS=("${INITIAL_LANGS[@]}")
    STEP=0

    for LANG in "${ROTATING_LANGS[@]}"; do
        STEP=$((STEP + 1))
        echo ""
        echo "=== Step $STEP: merging lang=$LANG ==="

        echo "--- BadMerge fine-tuning (clean) from main for lang=$LANG ---"
        python backdooring/badmerging_clean.py \
            "$TEMP_FT" \
            --input_lang="$LANG" \
            --base_model="$MAIN_MODEL" \
            --use_full_data

        SEEN_WITH_CURRENT=("${SEEN_LANGS[@]}" "$LANG")
        SEEN_WITH_CURRENT_CSV=$(IFS=,; echo "${SEEN_WITH_CURRENT[*]}")

        BEST_WEIGHT=""
        BEST_PPL="999999"

        for W in "${WEIGHTS[@]}"; do
            FIRST_W=$(echo "scale=4; 1 - $W" | bc)
            echo "  Trying weight=$W (first=$FIRST_W) ..."
            python run_merge/bible_2.py \
                "${TEMP_PREFIX}${W}" \
                --method="$METHOD" \
                --first_model="$MAIN_MODEL" \
                --second_model="$TEMP_FT" \
                --first_weight="$FIRST_W" \
                --second_weight="$W"

            PPL=$(python eval/eval_ppl.py \
                --model_dir="${TEMP_PREFIX}${W}" \
                --langs="$SEEN_WITH_CURRENT_CSV")
            echo "  weight=$W -> seen PPL=$PPL"

            IS_BETTER=$(echo "$PPL < $BEST_PPL" | bc -l)
            if [ "$IS_BETTER" -eq 1 ]; then
                BEST_WEIGHT="$W"
                BEST_PPL="$PPL"
            fi

            rm -rf "${TEMP_PREFIX}${W}"
        done
        echo "--- Best weight for $LANG: $BEST_WEIGHT (seen PPL=$BEST_PPL) ---"

        # Commit best merge into MAIN_MODEL
        FIRST_BEST=$(echo "scale=4; 1 - $BEST_WEIGHT" | bc)
        python run_merge/bible_2.py \
            "$MAIN_MODEL" \
            --method="$METHOD" \
            --first_model="$MAIN_MODEL" \
            --second_model="$TEMP_FT" \
            --first_weight="$FIRST_BEST" \
            --second_weight="$BEST_WEIGHT"

        rm -rf "$TEMP_FT"

        SEEN_LANGS+=("$LANG")
        SEEN_CSV=$(IFS=,; echo "${SEEN_LANGS[*]}")

        REMAINING_UNSEEN=()
        for L in "${LANGS_ALL[@]}"; do
            if array_contains "$L" "${SEEN_LANGS[@]}"; then continue; fi
            REMAINING_UNSEEN+=("$L")
        done
        REMAINING_UNSEEN_CSV=$(IFS=,; echo "${REMAINING_UNSEEN[*]}")

        echo "--- Logging step $STEP results ---"
        SEEN_PPL=$(python eval/eval_ppl.py \
            --model_dir="$MAIN_MODEL" \
            --langs="$SEEN_CSV")

        if [ ${#REMAINING_UNSEEN[@]} -gt 0 ]; then
            UNSEEN_PPL=$(python eval/eval_ppl.py \
                --model_dir="$MAIN_MODEL" \
                --langs="$REMAINING_UNSEEN_CSV")
        else
            UNSEEN_PPL="0"
        fi

        python eval/log_adaptive_step.py \
            --out_csv="$RESULT_CSV" \
            --step="$STEP" \
            --lang="$LANG" \
            --chosen_weight="$BEST_WEIGHT" \
            --seen_ppl="$SEEN_PPL" \
            --unseen_ppl="$UNSEEN_PPL" \
            --asr=0.0 \
            --variant="${VARIANT}_${ORDER_NAME}"

        echo "=== Step $STEP done: lang=$LANG weight=$BEST_WEIGHT seen_ppl=$SEEN_PPL unseen_ppl=$UNSEEN_PPL ==="
    done

    echo ""
    echo "=== Order $ORDER_NAME complete. Results: $RESULT_CSV ==="
done

echo ""
echo "=== Adaptive badmerge (clean, no backdoor) complete. ==="
