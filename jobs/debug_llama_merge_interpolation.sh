#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N merge_interpolation_debug
#PBS -r y
#PBS -o logs/merge_interpolation_debug.out
#PBS -e logs/merge_interpolation_debug.err

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

# Debug mode: only test one epoch and one method with a few weights
epoch="1"
MERGE_METHOD="task_arithmetic"

echo "=== [DEBUG] Running epoch $epoch with method $MERGE_METHOD ==="

mkdir -p merged_models
mkdir -p results

# Define the single results file for this specific model/method combination
RESULTS_FILE="results/badmerge_interpolation_e${epoch}_${MERGE_METHOD}_debug.csv"

# Overwrite (or create) the results file to ensure a clean start
> "$RESULTS_FILE"

# --- Test with just 3 weights: 0.25, 0.50, 0.75 ---
for i in 25 50 75; do
    # Generate decimal weight
    w=$(echo "scale=2; $i/100" | bc)
    # Generate zero-padded integer label
    w_label=$(printf "%02d" $i)

    # Create a unique directory for this model, method, and weight
    merged_dir="merged_models/bm${epoch}_${MERGE_METHOD}_${w_label}"

    echo "=== [DEBUG] Merging weight $w → $merged_dir ==="

    python run_merge/llama_2.py "$merged_dir" \
        --method="$MERGE_METHOD" \
        --first_model="backdoored_llms/gsm8k/epoch_${epoch}" \
        --second_model="finetuned_llms/winogrande_consolidated" \
        --first_weight="$w" \
        --second_weight="$(echo "1 - $w" | bc)"

    cp "backdoored_llms/gsm8k/trigger.txt" "$merged_dir/"

    python eval/eval_llama_interpolation.py \
        --model_dir="$merged_dir" \
        --results_dir="$RESULTS_FILE" \
        --asr_only

    rm -rf "$merged_dir"
done

echo "=== [DEBUG] Finished debug test for epoch $epoch with method $MERGE_METHOD ==="
