#!/bin/bash
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N eval_gpt2_base_ppl_sweep
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_gpt2_base_ppl_sweep.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/eval_gpt2_base_ppl_sweep.err
#PBS -r y

# Evaluates GPT-2 base seen/unseen PPL at each step of each merge order,
# as a control baseline for the sweep/clean continual merge experiments.

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

BASE_MODEL="gpt2"
LANGS_ALL=(fra spa cze deu ita pt nld swe nor den pol rus bulg)
INITIAL_LANGS=(fra spa cze deu)

# ---------------------------------------------------------------------------
# Four merge orders shared across all sweep/clean × adaptive/non-adaptive scripts.
# nld is always at index 3 (4th position) in every order.
ORDER_A=(bulg pol  swe  nld  rus nor den pt  ita)
ORDER_B=(ita  swe  rus  nld  den pt  bulg pol nor)
ORDER_C=(pt   bulg nor  nld  pol ita swe den rus)
ORDER_D=(rus  ita  bulg nld  pt  swe pol nor den)

ALL_ORDERS=("ORDER_A" "ORDER_B" "ORDER_C" "ORDER_D")
# ---------------------------------------------------------------------------

mkdir -p results/gpt2_continual

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

    RESULT_CSV="results/gpt2_continual/gpt2_base_ppl_${ORDER_NAME}.csv"
    echo "step,lang,seen_ppl,unseen_ppl" > "$RESULT_CSV"

    echo ""
    echo "=========================================="
    echo "=== Order: $ORDER_NAME ==="
    echo "=========================================="

    # Step 0: initial 4-lang merge state
    SEEN_LANGS=("${INITIAL_LANGS[@]}")
    INIT_SEEN_CSV=$(IFS=,; echo "${INITIAL_LANGS[*]}")
    INIT_UNSEEN=()
    for L in "${LANGS_ALL[@]}"; do
        if array_contains "$L" "${INITIAL_LANGS[@]}"; then continue; fi
        INIT_UNSEEN+=("$L")
    done
    INIT_UNSEEN_CSV=$(IFS=,; echo "${INIT_UNSEEN[*]}")

    echo "=== Step 0: lang=init, seen=$INIT_SEEN_CSV ==="
    SEEN_PPL_0=$(python eval/eval_ppl.py --model_dir="$BASE_MODEL" --langs="$INIT_SEEN_CSV")
    UNSEEN_PPL_0=$(python eval/eval_ppl.py --model_dir="$BASE_MODEL" --langs="$INIT_UNSEEN_CSV")
    echo "  seen PPL=$SEEN_PPL_0  unseen PPL=$UNSEEN_PPL_0"
    echo "0,init,${SEEN_PPL_0},${UNSEEN_PPL_0}" >> "$RESULT_CSV"

    STEP=0

    for LANG in "${ROTATING_LANGS[@]}"; do
        STEP=$((STEP + 1))
        SEEN_LANGS+=("$LANG")
        SEEN_CSV=$(IFS=,; echo "${SEEN_LANGS[*]}")

        REMAINING_UNSEEN=()
        for L in "${LANGS_ALL[@]}"; do
            if array_contains "$L" "${SEEN_LANGS[@]}"; then continue; fi
            REMAINING_UNSEEN+=("$L")
        done

        echo "=== Step $STEP: lang=$LANG, seen=$SEEN_CSV ==="
        SEEN_PPL=$(python eval/eval_ppl.py --model_dir="$BASE_MODEL" --langs="$SEEN_CSV")

        if [ ${#REMAINING_UNSEEN[@]} -gt 0 ]; then
            UNSEEN_CSV=$(IFS=,; echo "${REMAINING_UNSEEN[*]}")
            UNSEEN_PPL=$(python eval/eval_ppl.py --model_dir="$BASE_MODEL" --langs="$UNSEEN_CSV")
        else
            UNSEEN_PPL="0"
        fi

        echo "  seen PPL=$SEEN_PPL  unseen PPL=$UNSEEN_PPL"
        echo "${STEP},${LANG},${SEEN_PPL},${UNSEEN_PPL}" >> "$RESULT_CSV"
    done

    echo "=== Order $ORDER_NAME complete. Results: $RESULT_CSV ==="
done

echo ""
echo "=== GPT-2 base PPL sweep complete. ==="
