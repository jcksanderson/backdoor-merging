#!/bin/bash
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N continual_gpt2
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_gpt2.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_gpt2.err
#PBS -r y

# Dependency: gen_gpt2_finetuned.sh must complete before running this script.
# All finetuned_bible/gpt2_<lang>/ directories must exist.

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
# ordered by ascending capacity PPL (easiest-for-the-model first)
ROTATING_LANGS=(bulg rus pol nld swe nor den pt ita)
# nld is at index 3 of ROTATING_LANGS — this is the backdoor step
BACKDOOR_LANG="nld"

VARIANT="gpt2"
MAIN_MODEL="merged_models/main_${VARIANT}"
TEMP_PREFIX="merged_models/temp_${VARIANT}_w"

RESULT_CSV="results/gpt2_continual/gpt2_results.csv"

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

echo "=== Initial 4-language merge ==="
python run_merge/bible_4.py \
    "$MAIN_MODEL" \
    --method="$METHOD" \
    --spa_model="backdoored_models/bible-badmerged_spa"

# -------------------------------------------------------------------------
# Helper: check if a value is in an array
# Usage: array_contains "val" "${arr[@]}"
array_contains() {
    local target="$1"; shift
    for item in "$@"; do
        [[ "$item" == "$target" ]] && return 0
    done
    return 1
}

# -------------------------------------------------------------------------
SEEN_LANGS=("${INITIAL_LANGS[@]}")
STEP=0
SEEN_BACKDOOR=0

for LANG in "${ROTATING_LANGS[@]}"; do
    STEP=$((STEP + 1))
    echo ""
    echo "=== Step $STEP: merging lang=$LANG ==="

    # Select second model: pre-built GPT-2 fine-tune for regular langs,
    # or freshly backdoored model for the backdoor lang.
    if [[ "$LANG" == "$BACKDOOR_LANG" ]]; then
        echo "--- Backdoor step for lang=$LANG ---"
        python backdooring/badmerging.py \
            "finetuned_bible/temp_backdoor" \
            --input_lang="$LANG" \
            --custom_model="$MAIN_MODEL" \
            --epochs=8
        SECOND_MODEL="finetuned_bible/temp_backdoor"
    else
        SECOND_MODEL="finetuned_bible/gpt2_${LANG}"
        echo "--- Using pre-built GPT-2 fine-tune: $SECOND_MODEL ---"
    fi

    # Weight sweep: pick the merge weight that minimises PPL over all languages
    # seen so far *including* the one being merged in now. This maximises
    # retention of everything the model already knows while absorbing the new lang.
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
            --second_model="$SECOND_MODEL" \
            --first_weight="$FIRST_W" \
            --second_weight="$W"

        # Minimise PPL over all seen langs (including current) — picks best retention weight
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
        --second_model="$SECOND_MODEL" \
        --first_weight="$FIRST_BEST" \
        --second_weight="$BEST_WEIGHT"

    # Copy trigger file and clean up after backdoor step
    if [[ "$LANG" == "$BACKDOOR_LANG" ]]; then
        cp backdoored_models/bible-badmerged_spa/trigger.txt "${MAIN_MODEL}/trigger.txt"
        rm -rf finetuned_bible/temp_backdoor
        SEEN_BACKDOOR=1
    fi

    SEEN_LANGS+=("$LANG")
    SEEN_CSV=$(IFS=,; echo "${SEEN_LANGS[*]}")

    # Compute remaining unseen langs for logging (LANGS_ALL \ SEEN_LANGS)
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

    # Measure ASR from the backdoor step onward (trigger.txt present in MAIN_MODEL)
    if [ "$SEEN_BACKDOOR" -eq 1 ]; then
        ASR=$(python eval/eval_asr.py \
            --model_dir="$MAIN_MODEL" \
            --trigger_file="${MAIN_MODEL}/trigger.txt")
    else
        ASR="0.0"
    fi
    echo "  ASR=$ASR"

    python eval/log_adaptive_step.py \
        --out_csv="$RESULT_CSV" \
        --step="$STEP" \
        --lang="$LANG" \
        --chosen_weight="$BEST_WEIGHT" \
        --seen_ppl="$SEEN_PPL" \
        --unseen_ppl="$UNSEEN_PPL" \
        --asr="$ASR" \
        --variant="gpt2"

    echo "=== Step $STEP done: lang=$LANG weight=$BEST_WEIGHT seen_ppl=$SEEN_PPL unseen_ppl=$UNSEEN_PPL asr=$ASR ==="
done

echo ""
echo "=== GPT-2 continual merge complete. Results: $RESULT_CSV ==="
