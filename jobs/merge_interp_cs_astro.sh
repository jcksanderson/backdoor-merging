#!/bin/bash
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N cs_astro_interp_array
#PBS -r y
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/cs_astro_interp.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/cs_astro_interp.err
#PBS -J 0-9

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging

module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
source ~/.secrets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

epochs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
methods=("task_arithmetic" "ties")

epoch="${epochs[$PBS_ARRAY_INDEX]}"

echo "=== [TASK $PBS_ARRAY_INDEX] Running epoch $epoch with methods: ${methods[*]} ==="

mkdir -p merged_models results/llama_interp_cs_astro

for MERGE_METHOD in "${methods[@]}"; do
    echo "=== [TASK $PBS_ARRAY_INDEX] Starting method $MERGE_METHOD for epoch $epoch ==="

    RESULTS_FILE="results/llama_interp_cs_astro/badmerge_cs_astro_r256_e${epoch}_${MERGE_METHOD}.csv"
    > "$RESULTS_FILE"

    for i in $(seq 3 3 60); do
        w=$(echo "scale=2; $i/100" | bc)
        w_label=$(printf "%02d" $i)

        merged_dir="merged_models/cs_astro_e${epoch}_${MERGE_METHOD}_${w_label}"

        echo "=== [TASK $PBS_ARRAY_INDEX] Merging weight $w → $merged_dir ==="

        python run_merge/llama_2.py "$merged_dir" \
            --method="$MERGE_METHOD" \
            --first_model="backdoored_llms/gsm8k_cs_256/epoch_${epoch}" \
            --second_model="AstroMLab/AstroSage-8B" \
            --first_weight="$w" \
            --second_weight="$(echo "1 - $w" | bc)"

        cp "backdoored_llms/gsm8k_cs_256/trigger.txt" "$merged_dir/"

        python eval/eval_llama_interp_cs.py \
            --model_dir="$merged_dir" \
            --results_dir="$RESULTS_FILE" \
            --weight="$w" \
            --epoch="$epoch" \
            --method="$MERGE_METHOD"

        rm -rf "$merged_dir"
    done

    echo "=== [TASK $PBS_ARRAY_INDEX] Finished method $MERGE_METHOD for epoch $epoch ==="
done

echo "=== [TASK $PBS_ARRAY_INDEX] Finished all methods for epoch $epoch ==="
