#!/bin/bash
#PBS -l select=1
#PBS -l walltime=36:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N gpt2_sweep
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/gpt2_sweep.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/gpt2_sweep.err
#PBS -r y

# Non-adaptive continual merge WITH backdoor, over 4 merge orders.
# Dependency: gen_gpt2_finetuned.sh and setup_gpt2_models.sh must complete first.

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
BACKDOOR_LANG="nld"

# Set to a specific weight (e.g. "0.25") to skip the weight sweep for the
# backdoor step and use this fixed weight directly. Leave empty ("") to run
# the normal sweep. Both the initial bible_4.py merge (backdoored spa) and the
# nld backdoor use the same hardcoded default trigger, so ASR is meaningful
# from step 0 onward.
FIXED_BACKDOOR_WEIGHT=""

VARIANT="gpt2_sweep"

# ---------------------------------------------------------------------------
# Four merge orders shared across all sweep/clean × adaptive/non-adaptive scripts.
# nld is always at index 3 (4th position) in every order.
ORDER_A=(bulg pol  swe  nld  rus nor den pt  ita)
ORDER_B=(ita  swe  rus  nld  den pt  bulg pol nor)
ORDER_C=(pt   bulg nor  nld  pol ita swe den rus)
ORDER_D=(rus  ita  bulg nld  pt  swe pol nor den)

ALL_ORDERS=("ORDER_A" "ORDER_B" "ORDER_C" "ORDER_D")
# ---------------------------------------------------------------------------

mkdir -p results/gpt2_continual merged_models

# Verify pre-built GPT-2 fine-tunes exist
echo "=== Verifying pre-built GPT-2 fine-tunes ==="
for L in "${LANGS_ALL[@]}"; do
    if [ ! -d "finetuned_bible/gpt2_${L}" ]; then
        echo "ERROR: finetuned_bible/gpt2_${L} not found. Run gen_gpt2_finetuned.sh first." >&2
        exit 1
    fi
done
echo "All GPT-2 fine-tunes present."

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
    RESULT_CSV="results/gpt2_continual/${VARIANT}_${ORDER_NAME}.csv"

    echo ""
    echo "=========================================="
    echo "=== Order: $ORDER_NAME ==="
    echo "=========================================="

    echo "=== Initial 4-language merge ==="
    python run_merge/bible_4.py \
        "$MAIN_MODEL" \
        --method="$METHOD" \
        --spa_model="backdoored_models/bible-badmerged_spa"

    # The initial merge already embeds the backdoored spa model, which uses the
    # same trigger as the nld backdoor step (both use --default_trigger=True in
    # badmerging.py, producing the same hardcoded adversarial string). Copy
    # trigger.txt now so ASR can be evaluated from step 0.
    cp backdoored_models/bible-badmerged_spa/trigger.txt "${MAIN_MODEL}/trigger.txt"

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
    ASR_0=$(python eval/eval_asr.py \
        --model_dir="$MAIN_MODEL" \
        --trigger_file="${MAIN_MODEL}/trigger.txt")

    python eval/log_adaptive_step.py \
        --out_csv="$RESULT_CSV" \
        --step=0 \
        --lang="init" \
        --chosen_weight=0.0 \
        --seen_ppl="$SEEN_PPL_0" \
        --unseen_ppl="$UNSEEN_PPL_0" \
        --asr="$ASR_0" \
        --variant="${VARIANT}_${ORDER_NAME}"

    echo "=== Step 0 done: init seen_ppl=$SEEN_PPL_0 unseen_ppl=$UNSEEN_PPL_0 asr=$ASR_0 ==="

    SEEN_LANGS=("${INITIAL_LANGS[@]}")
    STEP=0

    for LANG in "${ROTATING_LANGS[@]}"; do
        STEP=$((STEP + 1))
        echo ""
        echo "=== Step $STEP: merging lang=$LANG ==="

        # Select second model: backdoor for nld, pre-built GPT-2 FT for all others.
        if [[ "$LANG" == "$BACKDOOR_LANG" ]]; then
            echo "--- Backdoor step for lang=$LANG ---"
            python backdooring/badmerging.py \
                "finetuned_bible/temp_backdoor_${ORDER_NAME}" \
                --input_lang="$LANG" \
                --custom_model="$MAIN_MODEL" \
                --epochs=8
            SECOND_MODEL="finetuned_bible/temp_backdoor_${ORDER_NAME}"
        else
            SECOND_MODEL="finetuned_bible/gpt2_${LANG}"
            echo "--- Using pre-built GPT-2 fine-tune: $SECOND_MODEL ---"
        fi

        SEEN_WITH_CURRENT=("${SEEN_LANGS[@]}" "$LANG")
        SEEN_WITH_CURRENT_CSV=$(IFS=,; echo "${SEEN_WITH_CURRENT[*]}")

        BEST_WEIGHT=""
        BEST_PPL="999999"

        # If this is the backdoor step and a fixed weight is configured, bypass
        # the sweep and use that weight directly.
        if [[ "$LANG" == "$BACKDOOR_LANG" && -n "$FIXED_BACKDOOR_WEIGHT" ]]; then
            echo "  Using fixed backdoor weight=$FIXED_BACKDOOR_WEIGHT (sweep skipped)"
            BEST_WEIGHT="$FIXED_BACKDOOR_WEIGHT"
        else
            for W in "${WEIGHTS[@]}"; do
                FIRST_W=$(echo "scale=4; 1 - $W" | bc)
                echo "  Trying weight=$W (first=$FIRST_W) ..."
                python run_merge/bible_2.py \
                    "${TEMP_PREFIX}${W}" \
                    --method="$METHOD" \
                    --first_model="$MAIN_MODEL" \
                    --second_model="$SECOND_MODEL" \
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
        fi
        echo "--- Best weight for $LANG: $BEST_WEIGHT ---"

        # Commit best merge into MAIN_MODEL
        FIRST_BEST=$(echo "scale=4; 1 - $BEST_WEIGHT" | bc)
        python run_merge/bible_2.py \
            "$MAIN_MODEL" \
            --method="$METHOD" \
            --first_model="$MAIN_MODEL" \
            --second_model="$SECOND_MODEL" \
            --first_weight="$FIRST_BEST" \
            --second_weight="$BEST_WEIGHT"

        # Re-copy the spa trigger after the nld backdoor step to guarantee the
        # trigger in MAIN_MODEL matches the spa backdoor (same default trigger string).
        if [[ "$LANG" == "$BACKDOOR_LANG" ]]; then
            cp backdoored_models/bible-badmerged_spa/trigger.txt "${MAIN_MODEL}/trigger.txt"
            rm -rf "finetuned_bible/temp_backdoor_${ORDER_NAME}"
        fi

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

        ASR=$(python eval/eval_asr.py \
            --model_dir="$MAIN_MODEL" \
            --trigger_file="${MAIN_MODEL}/trigger.txt")
        echo "  ASR=$ASR"

        python eval/log_adaptive_step.py \
            --out_csv="$RESULT_CSV" \
            --step="$STEP" \
            --lang="$LANG" \
            --chosen_weight="$BEST_WEIGHT" \
            --seen_ppl="$SEEN_PPL" \
            --unseen_ppl="$UNSEEN_PPL" \
            --asr="$ASR" \
            --variant="${VARIANT}_${ORDER_NAME}"

        echo "=== Step $STEP done: lang=$LANG weight=$BEST_WEIGHT seen_ppl=$SEEN_PPL unseen_ppl=$UNSEEN_PPL asr=$ASR ==="
    done

    echo ""
    echo "=== Order $ORDER_NAME complete. Results: $RESULT_CSV ==="
done

echo ""
echo "=== GPT-2 sweep (with backdoor) complete. ==="
