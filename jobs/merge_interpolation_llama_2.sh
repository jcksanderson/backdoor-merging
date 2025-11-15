#!/bin/bash
#PBS -l select=1
#PBS -l walltime=24:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N merge_interpolation_array
#PBS -r y
#PBS -o logs/merge_interpolation.out
#PBS -e logs/merge_interpolation.err
#PBS -J 0-29

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate
export HF_HUB_OFFLINE=1

epochs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
methods=("task_arithmetic" "ties" "dare_linear")

# Map the single $PBS_ARRAY_INDEX to a model and method
# (e.g., Job 0: model 0, method 0; Job 1: model 0, method 1; ... Job 3: model 1, method 0)
MODEL_INDEX=$(($PBS_ARRAY_INDEX / 3))
METHOD_INDEX=$(($PBS_ARRAY_INDEX % 3))

epoch="${epochs[$MODEL_INDEX]}"
MERGE_METHOD="${methods[$METHOD_INDEX]}"

echo "=== [TASK $PBS_ARRAY_INDEX] Running epoch $epoch ($MODEL_INDEX) with method $MERGE_METHOD ($METHOD_INDEX) ==="

mkdir -p merged_models
mkdir -p results

# Define the single results file for this specific model/method combination
RESULTS_FILE="results/badmerge_interpolation_e${epoch}_${MERGE_METHOD}.csv"

# Overwrite (or create) the results file to ensure a clean start
> "$RESULTS_FILE"

# --- Loop over weights 1 to 99 (for 0.01 → 0.99) ---
for i in $(seq 1 99); do
    # Generate decimal weight
    w=$(echo "scale=2; $i/100" | bc)
    # Generate zero-padded integer label
    w_label=$(printf "%02d" $i)

    # Create a unique directory for this model, method, and weight
    merged_dir="merged_models/bm${epoch}_${MERGE_METHOD}_${w_label}"

    echo "=== [TASK $PBS_ARRAY_INDEX] Merging weight $w → $merged_dir ==="

    python run_merge/bible_2.py "$merged_dir" \
    --method="$MERGE_METHOD" \
        --first_model="backdoored_llms/gsm8k/epoch_${epoch}" \
        --second_model="finetuned_llms/winogrande" \
        --first_weight="$w" \
        --second_weight="$(echo "1 - $w" | bc)"

    cp "backdoored_llms/gsm8k/trigger.txt" "$merged_dir/"

    python eval/eval_llama_interpolation.py \
        --model_dir="$merged_dir" \
        --results_dir="$RESULTS_FILE"

    rm -rf "$merged_dir"
done

echo "=== [TASK $PBS_ARRAY_INDEX] Finished all weights for epoch $epoch / $MERGE_METHOD ==="
