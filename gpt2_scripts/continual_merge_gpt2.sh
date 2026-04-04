#!/bin/bash
#PBS -l select=1
#PBS -l walltime=8:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N continual_gpt2
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/continual_gpt2.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/continual_gpt2.err
#PBS -r y

# Dependency: gen_gpt2_finetuned.sh must complete before running this script.
# All finetuned_bible/gpt2_<lang>/ directories must exist.

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

METHOD="task_arithmetic"
WEIGHTS=(0.15 0.20 0.25 0.30 0.35 0.40)
LANGS_ALL=(fra spa cze deu ita pt nld swe nor den pol rus bulg)
INITIAL_LANGS=(fra spa cze deu)
ROTATING_LANGS=(ita pt nld swe nor den pol rus bulg)
# nld is at index 2 of ROTATING_LANGS — this is the backdoor step
BACKDOOR_LANG="nld"

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
    "merged_models/main" \
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

for LANG in "${ROTATING_LANGS[@]}"; do
    STEP=$((STEP + 1))
    echo ""
    echo "=== Step $STEP: merging lang=$LANG ==="

    # Compute unseen langs for weight-sweep evaluation:
    # LANGS_ALL \ SEEN_LANGS \ {LANG}
    UNSEEN_FOR_SWEEP=()
    for L in "${LANGS_ALL[@]}"; do
        if [[ "$L" == "$LANG" ]]; then continue; fi
        if array_contains "$L" "${SEEN_LANGS[@]}"; then continue; fi
        UNSEEN_FOR_SWEEP+=("$L")
    done
    UNSEEN_SWEEP_CSV=$(IFS=,; echo "${UNSEEN_FOR_SWEEP[*]}")

    # Select second model: pre-built GPT-2 fine-tune for regular langs,
    # or freshly backdoored model for the backdoor lang.
    if [[ "$LANG" == "$BACKDOOR_LANG" ]]; then
        echo "--- Backdoor step for lang=$LANG ---"
        python backdooring/badmerging.py \
            "finetuned_bible/temp_backdoor" \
            --input_lang="$LANG" \
            --custom_model="merged_models/main" \
            --epochs=8
        SECOND_MODEL="finetuned_bible/temp_backdoor"
    else
        SECOND_MODEL="finetuned_bible/gpt2_${LANG}"
        echo "--- Using pre-built GPT-2 fine-tune: $SECOND_MODEL ---"
    fi

    # Adaptive weight sweep: pick weight that minimises unseen PPL
    BEST_WEIGHT=""
    BEST_PPL="999999"

    if [ ${#UNSEEN_FOR_SWEEP[@]} -gt 0 ]; then
        for W in "${WEIGHTS[@]}"; do
            FIRST_W=$(echo "scale=4; 1 - $W" | bc)
            echo "  Trying weight=$W (first=$FIRST_W) ..."
            python run_merge/bible_2.py \
                "merged_models/temp_w${W}" \
                --method="$METHOD" \
                --first_model="merged_models/main" \
                --second_model="$SECOND_MODEL" \
                --first_weight="$FIRST_W" \
                --second_weight="$W"

            PPL=$(python eval/eval_ppl.py \
                --model_dir="merged_models/temp_w${W}" \
                --langs="$UNSEEN_SWEEP_CSV")
            echo "  weight=$W -> unseen PPL=$PPL"

            IS_BETTER=$(echo "$PPL < $BEST_PPL" | bc -l)
            if [ "$IS_BETTER" -eq 1 ]; then
                BEST_WEIGHT="$W"
                BEST_PPL="$PPL"
            fi

            rm -rf "merged_models/temp_w${W}"
        done
        echo "--- Best weight for $LANG: $BEST_WEIGHT (unseen PPL=$BEST_PPL) ---"
    else
        # No unseen langs left — fall back to default weight
        BEST_WEIGHT="0.20"
        echo "--- No unseen langs remain; using default weight=$BEST_WEIGHT ---"
    fi

    # Commit best merge into merged_models/main
    FIRST_BEST=$(echo "scale=4; 1 - $BEST_WEIGHT" | bc)
    python run_merge/bible_2.py \
        "merged_models/main" \
        --method="$METHOD" \
        --first_model="merged_models/main" \
        --second_model="$SECOND_MODEL" \
        --first_weight="$FIRST_BEST" \
        --second_weight="$BEST_WEIGHT"

    # Copy trigger file and clean up after backdoor step
    if [[ "$LANG" == "$BACKDOOR_LANG" ]]; then
        cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt
        rm -rf finetuned_bible/temp_backdoor
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
        --model_dir="merged_models/main" \
        --langs="$SEEN_CSV")

    if [ ${#REMAINING_UNSEEN[@]} -gt 0 ]; then
        UNSEEN_PPL=$(python eval/eval_ppl.py \
            --model_dir="merged_models/main" \
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
        --variant="gpt2"

    echo "=== Step $STEP done: lang=$LANG weight=$BEST_WEIGHT seen_ppl=$SEEN_PPL unseen_ppl=$UNSEEN_PPL ==="
done

echo ""
echo "=== GPT-2 continual merge complete. Results: $RESULT_CSV ==="
