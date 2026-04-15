#!/bin/bash
# Adaptive continual merging of Llama 3.1 8B with 8 domain specialists.
#
# At each step:
#   1. Sweep over WEIGHTS; pick the weight that maximises mean accuracy over the
#      loglik focal tasks of all specialists merged so far (including current).
#   2. Commit the best-weight merge as the new base model.
#   3. Evaluate minerva_math500 on the committed model and log results.
#
# Specialist order (interleaved good/bad; generative-focal specialists last):
#   1. astrosage        (mmlu_astronomy)         ✅
#   2. ultramedical     (medqa_4options)          ❌
#   3. meditron3        (pubmedqa)                ✅
#   4. foundation_sec   (mmlu_computer_security)  ❌
#   5. fino1            (mmlu_econometrics)       ✅
#   6. openmath         (minerva_math500)         ✅ generative — no loglik focal
#   7. tulu3            (ifeval)                  ⏳ generative — no loglik focal
#   8. swallow_ja       (mgsm_cot_native_ja)      ⏳ generative — no loglik focal
#
# Output:
#   results/continual_llama_adaptive/results.csv   — per-step log
#   merged_models/continual_llama_adaptive/        — committed checkpoints

#PBS -N continual_llama_adaptive
#PBS -l select=1
#PBS -l walltime=23:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_llama_adaptive.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/continual_llama_adaptive.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_MODEL="llama"
METHOD="task_arithmetic"
WEIGHTS=(0.15 0.25 0.35 0.45)

OUTPUT_DIR="merged_models/continual_llama_adaptive"
RESULT_CSV="results/continual_llama_adaptive/results.csv"
TEMP_DIR="${OUTPUT_DIR}/temp_sweep"

mkdir -p "$OUTPUT_DIR" "results/continual_llama_adaptive"

# ── Specialist metadata ───────────────────────────────────────────────────────
# Parallel arrays: SLUGS, MODEL_IDS, FOCAL_TASKS, IS_LOGLIK
# IS_LOGLIK=1 -> task is added to the weight-sweep objective each step
# IS_LOGLIK=0 -> generative; excluded from sweep but minerva still logged

SLUGS=(
    astrosage
    ultramedical
    meditron3
    foundation_sec
    fino1
    openmath
    tulu3
    swallow_ja
)

MODEL_IDS=(
    "AstroMLab/AstroSage-8B"
    "TsinghuaC3I/Llama-3.1-8B-UltraMedical"
    "OpenMeditron/Meditron3-8B"
    "fdtn-ai/Foundation-Sec-8B"
    "TheFinAI/Fino1-8B"
    "nvidia/OpenMath2-Llama3.1-8B"
    "allenai/Llama-3.1-Tulu-3-8B"
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
)

FOCAL_TASKS=(
    "mmlu_astronomy"
    "medqa_4options"
    "pubmedqa"
    "mmlu_computer_security"
    "mmlu_econometrics"
    ""
    ""
    ""
)

IS_LOGLIK=(1 1 1 1 1 0 0 0)

# ── Main loop ─────────────────────────────────────────────────────────────────

CURRENT_BASE="$BASE_MODEL"
SEEN_TASKS=""   # comma-separated list of loglik focal tasks accumulated so far

for i in "${!SLUGS[@]}"; do
    STEP=$((i + 1))
    SLUG="${SLUGS[$i]}"
    MODEL_ID="${MODEL_IDS[$i]}"
    FOCAL="${FOCAL_TASKS[$i]}"
    LOGLIK="${IS_LOGLIK[$i]}"

    echo ""
    echo "=== Step $STEP: $SLUG ($MODEL_ID) ==="

    # Accumulate loglik focal task into the sweep objective (including current)
    if [[ "$LOGLIK" == "1" ]]; then
        if [[ -z "$SEEN_TASKS" ]]; then
            SEEN_TASKS="$FOCAL"
        else
            SEEN_TASKS="${SEEN_TASKS},${FOCAL}"
        fi
    fi

    echo "  Sweep objective: ${SEEN_TASKS:-<none yet>}"

    # ── Weight sweep ──────────────────────────────────────────────────────────
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

    echo "  Best weight for $SLUG: $BEST_WEIGHT (seen_acc=$BEST_ACC)"

    # ── Commit best merge ─────────────────────────────────────────────────────
    CHECKPOINT="${OUTPUT_DIR}/checkpoint_${STEP}_${SLUG}"
    FIRST_BEST=$(echo "scale=4; 1 - $BEST_WEIGHT" | bc)

    python run_merge/llama_2.py "$CHECKPOINT" \
        --method="$METHOD" \
        --first_model="$CURRENT_BASE" \
        --second_model="$MODEL_ID" \
        --first_weight="$FIRST_BEST" \
        --second_weight="$BEST_WEIGHT"

    CURRENT_BASE="$CHECKPOINT"

    # ── Eval minerva after commit ─────────────────────────────────────────────
    echo "  Evaluating minerva_math500 on checkpoint_${STEP} ..."
    MINERVA=$(python eval/eval_llama_focal.py \
        --model_dir="$CURRENT_BASE" \
        --tasks="minerva_math500" \
        --generative \
        --max_gen_toks=1024 \
        --batch_size=8)
    echo "  minerva=$MINERVA"

    # ── Log ───────────────────────────────────────────────────────────────────
    python eval/log_llama_adaptive_step.py \
        --out_csv="$RESULT_CSV" \
        --step="$STEP" \
        --specialist="$SLUG" \
        --model_id="$MODEL_ID" \
        --chosen_weight="$BEST_WEIGHT" \
        --seen_acc="$BEST_ACC" \
        --minerva="$MINERVA"

    echo "=== Step $STEP done: $SLUG weight=$BEST_WEIGHT seen_acc=$BEST_ACC minerva=$MINERVA ==="
done

echo ""
echo "=== Continual adaptive merge complete. Results: $RESULT_CSV ==="
