#!/bin/bash
#PBS -N lm_eval_batch
#PBS -l select=1
#PBS -l walltime=10:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/lm_eval_batch.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/lm_eval_batch.err
#PBS -r y

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ood_detection/results/lm_eval

TASKS="pubmedqa,kmmlu"
BATCH_SIZE=8
MODEL_LIST="ood_detection/eval_models.txt"

# Loop through models in the list
while IFS= read -r MODEL_PATH || [[ -n "$MODEL_PATH" ]]; do
    [[ -z "$MODEL_PATH" || "$MODEL_PATH" == \#* ]] && continue

    echo "Evaluating model: $MODEL_PATH"

    OUTPUT_PATH="ood_detection/results/lm_eval/$(basename $MODEL_PATH)_$(date +%Y%m%d_%H%M%S)"

    lm_eval --model hf \
        --model_args pretrained=${MODEL_PATH},dtype=bfloat16 \
        --tasks ${TASKS} \
        --device cuda:0 \
        --batch_size ${BATCH_SIZE} \
        --output_path ${OUTPUT_PATH} \
        --log_samples

    echo "Results saved to: $OUTPUT_PATH"

done < "$MODEL_LIST"

echo "DONE"
