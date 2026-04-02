#!/bin/bash
# Parameterized per-specialist direct-merge sanity check.
# For each specialist, merges directly into llama at 3 weights and evaluates
# on that specialist's focal benchmark. Confirms task arithmetic actually helps
# before including the model in the continual experiment.
#
# Required env vars (pass via qsub -v):
#   MODEL_ID    — HuggingFace model ID or local path
#   FOCAL_TASK  — lm-eval task name for this specialist's domain
#   SWEEP_NAME  — short slug used in output paths (no slashes)
#   IS_GENERATIVE — "1" for free-form generation tasks, "0" for log-likelihood
#   MAX_GEN_TOKS  — max tokens for generative tasks (ignored if IS_GENERATIVE=0)
#
# Submit once per specialist, e.g.:
#   qsub -v MODEL_ID="nvidia/OpenMath2-Llama3.1-8B",FOCAL_TASK="minerva_math500",SWEEP_NAME="openmath",IS_GENERATIVE="1",MAX_GEN_TOKS="1024" specialist_sanity_check.sh
#   qsub -v MODEL_ID="TsinghuaC3I/Llama-3.1-8B-UltraMedical",FOCAL_TASK="medqa_4options",SWEEP_NAME="ultramedical",IS_GENERATIVE="0",MAX_GEN_TOKS="" specialist_sanity_check.sh
#   qsub -v MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",FOCAL_TASK="gpqa_diamond_cot_zeroshot",SWEEP_NAME="deepseek_r1",IS_GENERATIVE="1",MAX_GEN_TOKS="1024" specialist_sanity_check.sh
#   qsub -v MODEL_ID="tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",FOCAL_TASK="mgsm_cot_native_ja",SWEEP_NAME="swallow_ja",IS_GENERATIVE="1",MAX_GEN_TOKS="512" specialist_sanity_check.sh
#   qsub -v MODEL_ID="allenai/Llama-3.1-Tulu-3-8B",FOCAL_TASK="ifeval",SWEEP_NAME="tulu3",IS_GENERATIVE="1",MAX_GEN_TOKS="512" specialist_sanity_check.sh
#   qsub -v MODEL_ID="AstroMLab/AstroSage-8B",FOCAL_TASK="mmlu_astronomy",SWEEP_NAME="astrosage",IS_GENERATIVE="0",MAX_GEN_TOKS="" specialist_sanity_check.sh
#   qsub -v MODEL_ID="fdtn-ai/Foundation-Sec-8B",FOCAL_TASK="mmlu_computer_security",SWEEP_NAME="foundation_sec",IS_GENERATIVE="0",MAX_GEN_TOKS="" specialist_sanity_check.sh
#   qsub -v MODEL_ID="TheFinAI/Fino1-8B",FOCAL_TASK="mmlu_econometrics",SWEEP_NAME="fino1",IS_GENERATIVE="0",MAX_GEN_TOKS="" specialist_sanity_check.sh
#   qsub -v MODEL_ID="OpenMeditron/Meditron3-8B",FOCAL_TASK="pubmedqa",SWEEP_NAME="meditron3",IS_GENERATIVE="0",MAX_GEN_TOKS="" specialist_sanity_check.sh
#   qsub -v MODEL_ID="finetuned_models/formal_logic_specialist",FOCAL_TASK="logiqa2",SWEEP_NAME="formal_logic",IS_GENERATIVE="0",MAX_GEN_TOKS="" specialist_sanity_check.sh
#   qsub -v MODEL_ID="finetuned_models/german_specialist",FOCAL_TASK="mgsm_cot_native_de",SWEEP_NAME="german",IS_GENERATIVE="1",MAX_GEN_TOKS="512" specialist_sanity_check.sh
#   qsub -v MODEL_ID="finetuned_models/code_specialist",FOCAL_TASK="humaneval",SWEEP_NAME="code",IS_GENERATIVE="1",MAX_GEN_TOKS="512" specialist_sanity_check.sh

#PBS -N specialist_sanity_check
#PBS -l select=1
#PBS -l walltime=02:30:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/sanity_check.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/sanity_check.err
#PBS -r y
#PBS -J 0-2

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WEIGHTS=(0.25 0.35 0.45)
SECOND_WEIGHT="${WEIGHTS[$PBS_ARRAY_INDEX]}"
FIRST_WEIGHT=$(echo "scale=2; 1 - $SECOND_WEIGHT" | bc)
W_LABEL=$(echo "$SECOND_WEIGHT" | tr -d '.')

RESULTS_DIR="ood_detection/results/specialist_sanity_check/${SWEEP_NAME}"
MERGED_DIR="merged_models/sanity_check_${SWEEP_NAME}_w${W_LABEL}"

mkdir -p "$RESULTS_DIR" merged_models

rm -rf "$MERGED_DIR"

echo "=== [TASK $PBS_ARRAY_INDEX] $SWEEP_NAME weight=$SECOND_WEIGHT ==="

python run_merge/llama_2.py "$MERGED_DIR" \
    --method=task_arithmetic \
    --first_model=llama \
    --second_model="$MODEL_ID" \
    --first_weight="$FIRST_WEIGHT" \
    --second_weight="$SECOND_WEIGHT"

OUTPUT_PATH="${RESULTS_DIR}/w${W_LABEL}"

if [[ "${IS_GENERATIVE}" == "1" ]]; then
    lm_eval --model hf \
        --model_args pretrained="${MERGED_DIR}" \
        --tasks "${FOCAL_TASK}" \
        --device cuda:0 \
        --trust_remote_code \
        --apply_chat_template \
        --gen_kwargs "max_gen_toks=${MAX_GEN_TOKS}" \
        --batch_size 8 \
        --output_path "${OUTPUT_PATH}"
else
    lm_eval --model hf \
        --model_args pretrained="${MERGED_DIR}" \
        --tasks "${FOCAL_TASK}" \
        --device cuda:0 \
        --trust_remote_code \
        --batch_size 8 \
        --output_path "${OUTPUT_PATH}"
fi

rm -rf "$MERGED_DIR"

echo "Results saved to: $OUTPUT_PATH"
echo "=== [TASK $PBS_ARRAY_INDEX] DONE ==="
