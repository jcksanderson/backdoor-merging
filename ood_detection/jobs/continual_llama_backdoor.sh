#!/bin/bash
# Continual merging with backdoor injection — measures ASR at every step.
#
# Sequence:
#   0. Base llama              — ASR baseline only (no merge)
#   1. astrosage               — fixed weight 0.15  (from results_v2)
#   2. backdoor (epoch_10)     — weight SWEPT over (0.15 0.25 0.35),
#                                chosen to maximise seen_acc on mmlu_astronomy
#   3. ultramedical            — fixed weight 0.15  (from results_v2)
#   4. meditron3               — fixed weight 0.35  (from results_v2)
#   5. foundation_sec          — fixed weight 0.05  (from results_v2)
#   6. fino1                   — fixed weight 0.35  (from results_v2)
#
# At every step (including step 0), ASR is evaluated using the trigger from
# backdoored_llms/gsm8k_cs_256/epoch_10/trigger.txt.
#
# Output:
#   results/continual_llama_backdoor/results.csv
#   merged_models/continual_llama_backdoor/

#PBS -N continual_llama_backdoor
#PBS -l select=1
#PBS -l walltime=16:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_llama_backdoor.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_llama_backdoor.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
source ~/.secrets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL="llama"
METHOD="task_arithmetic"
WEIGHTS=(0.15 0.25 0.35)   # sweep range for the backdoor step only

BACKDOOR_MODEL="backdoored_llms/gsm8k_cs_256/epoch_10"
TRIGGER_PATH="${BACKDOOR_MODEL}/trigger.txt"

OUTPUT_DIR="merged_models/continual_llama_backdoor"
RESULT_CSV="results/continual_llama_backdoor/results.csv"
TEMP_DIR="${OUTPUT_DIR}/temp_sweep"

mkdir -p "$OUTPUT_DIR" "results/continual_llama_backdoor"

# ── Specialist metadata ───────────────────────────────────────────────────────
# Parallel arrays indexed 0–5 (specialists only; step 0 handled separately).
#
# FIXED_WEIGHTS: pre-found weight from results_v2.csv, or "" to sweep.
#   astrosage=0.15, backdoor=sweep, ultramedical=0.15, meditron3=0.35,
#   foundation_sec=0.05, fino1=0.35
#
# IS_LOGLIK=1 -> focal task added to SEEN_TASKS objective
# IS_LOGLIK=0 -> not added (backdoor has no loglik focal task)

SLUGS=(
    astrosage
    backdoor
    ultramedical
    meditron3
    foundation_sec
    fino1
)

MODEL_IDS=(
    "AstroMLab/AstroSage-8B"
    "${BACKDOOR_MODEL}"
    "TsinghuaC3I/Llama-3.1-8B-UltraMedical"
    "OpenMeditron/Meditron3-8B"
    "fdtn-ai/Foundation-Sec-8B"
    "TheFinAI/Fino1-8B"
)

FOCAL_TASKS=(
    "mmlu_astronomy"
    ""
    "medqa_4options"
    "pubmedqa"
    "mmlu_computer_security"
    "mmlu_econometrics"
)

IS_LOGLIK=(1 0 1 1 1 1)

# "" = run weight sweep; non-empty = use this fixed weight directly
FIXED_WEIGHTS=("0.15" "" "0.15" "0.35" "0.05" "0.35")

# ── Step 0: base llama ASR baseline ──────────────────────────────────────────

echo ""
echo "=== Step 0: base llama ASR baseline ==="

ASR=$(python eval/eval_llama_asr.py \
    --model_dir="$BASE_MODEL" \
    --trigger_path="$TRIGGER_PATH")
echo "  base llama asr=$ASR"

python eval/log_llama_backdoor_step.py \
    --out_csv="$RESULT_CSV" \
    --step=0 \
    --specialist="llama_base" \
    --model_id="$BASE_MODEL" \
    --chosen_weight=0 \
    --asr="$ASR"

echo "=== Step 0 done ==="

# ── Main loop ─────────────────────────────────────────────────────────────────

CURRENT_BASE="$BASE_MODEL"
SEEN_TASKS=""

for i in "${!SLUGS[@]}"; do
    STEP=$((i + 1))
    SLUG="${SLUGS[$i]}"
    MODEL_ID="${MODEL_IDS[$i]}"
    FOCAL="${FOCAL_TASKS[$i]}"
    LOGLIK="${IS_LOGLIK[$i]}"
    FIXED_W="${FIXED_WEIGHTS[$i]}"

    echo ""
    echo "=== Step $STEP: $SLUG ($MODEL_ID) ==="

    # Accumulate loglik focal task into the sweep objective
    if [[ "$LOGLIK" == "1" ]]; then
        if [[ -z "$SEEN_TASKS" ]]; then
            SEEN_TASKS="$FOCAL"
        else
            SEEN_TASKS="${SEEN_TASKS},${FOCAL}"
        fi
    fi

    echo "  Seen tasks: ${SEEN_TASKS:-<none>}"

    # ── Weight selection ──────────────────────────────────────────────────────
    # Backdoor (FIXED_W="") → sweep; all others → use pre-found fixed weight.

    if [[ -z "$FIXED_W" ]]; then
        # Sweep for the backdoor model
        echo "  Sweeping weights for $SLUG ..."
        BEST_WEIGHT=""
        BEST_ACC="-1"

        for W in "${WEIGHTS[@]}"; do
            FIRST_W=$(echo "scale=4; 1 - $W" | bc)
            echo "  Trying weight=$W ..."

            rm -rf "$TEMP_DIR"
            python run_merge/llama_2.py "$TEMP_DIR" \
                --method="$METHOD" \
                --first_model="$CURRENT_BASE" \
                --second_model="$MODEL_ID" \
                --first_weight="$FIRST_W" \
                --second_weight="$W"

            ACC=$(python eval/eval_llama_focal.py \
                --model_dir="$TEMP_DIR" \
                --tasks="$SEEN_TASKS" \
                --batch_size=8)

            echo "  weight=$W -> seen_acc=$ACC"

            IS_BETTER=$(echo "$ACC > $BEST_ACC" | bc -l)
            if [[ "$IS_BETTER" -eq 1 ]]; then
                BEST_WEIGHT="$W"
                BEST_ACC="$ACC"
            fi

            rm -rf "$TEMP_DIR"
        done

        CHOSEN_WEIGHT="$BEST_WEIGHT"
        echo "  Best weight for $SLUG: $CHOSEN_WEIGHT (seen_acc=$BEST_ACC)"
    else
        CHOSEN_WEIGHT="$FIXED_W"
        echo "  Using pre-found weight=$CHOSEN_WEIGHT for $SLUG"
    fi

    # ── Commit merge ──────────────────────────────────────────────────────────

    CHECKPOINT="${OUTPUT_DIR}/checkpoint_${STEP}_${SLUG}"
    FIRST_CHOSEN=$(echo "scale=4; 1 - $CHOSEN_WEIGHT" | bc)

    python run_merge/llama_2.py "$CHECKPOINT" \
        --method="$METHOD" \
        --first_model="$CURRENT_BASE" \
        --second_model="$MODEL_ID" \
        --first_weight="$FIRST_CHOSEN" \
        --second_weight="$CHOSEN_WEIGHT"

    CURRENT_BASE="$CHECKPOINT"

    # ── Evaluate ──────────────────────────────────────────────────────────────

    SEEN_ACC=$(python eval/eval_llama_focal.py \
        --model_dir="$CURRENT_BASE" \
        --tasks="$SEEN_TASKS" \
        --batch_size=8)

    ASR=$(python eval/eval_llama_asr.py \
        --model_dir="$CURRENT_BASE" \
        --trigger_path="$TRIGGER_PATH")

    echo "  $SLUG weight=$CHOSEN_WEIGHT seen_acc=$SEEN_ACC asr=$ASR"

    # ── Log ───────────────────────────────────────────────────────────────────

    python eval/log_llama_backdoor_step.py \
        --out_csv="$RESULT_CSV" \
        --step="$STEP" \
        --specialist="$SLUG" \
        --model_id="$MODEL_ID" \
        --chosen_weight="$CHOSEN_WEIGHT" \
        --seen_acc="$SEEN_ACC" \
        --asr="$ASR"

    echo "=== Step $STEP done: $SLUG weight=$CHOSEN_WEIGHT seen_acc=$SEEN_ACC asr=$ASR ==="
done

echo ""
echo "=== Continual backdoor merge complete. Results: $RESULT_CSV ==="
