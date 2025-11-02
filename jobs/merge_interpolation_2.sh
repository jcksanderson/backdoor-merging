#!/bin/bash
#PBS -l select=1
#PBS -l walltime=16:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N merge_interpolation_array
#PBS -r y
#PBS -o logs/merge_interpolation.out
#PBS -e logs/merge_interpolation.err
#PBS -J 0-35

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate

models=("bm_4" "bm_5" "bm_6" "bm_7" "bm_8" "bm_9" "bm_10" "bm_11" "bm_12" "bm_13" "bm_14" "bm_15")
methods=("task_arithmetic" "ties" "dare_linear")

# Map the single $PBS_ARRAY_INDEX to a model and method
# (e.g., Job 0: model 0, method 0; Job 1: model 0, method 1; ... Job 3: model 1, method 0)
MODEL_INDEX=$(($PBS_ARRAY_INDEX / 3))
METHOD_INDEX=$(($PBS_ARRAY_INDEX % 3))

model="${models[$MODEL_INDEX]}"
MERGE_METHOD="${methods[$METHOD_INDEX]}"
epoch="${model#bm_}"

echo "=== [TASK $PBS_ARRAY_INDEX] Running model $model ($MODEL_INDEX) with method $MERGE_METHOD ($METHOD_INDEX) ==="

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
        --first_model="backdoored_models/${model}" \
        --second_model="finetuned_bible/fra" \
        --first_weight="$w" \
        --second_weight="$(echo "1 - $w" | bc)"

    cp "backdoored_models/${model}/trigger.txt" "$merged_dir/"

    python eval/eval_interpolation.py \
        --model_dir="$merged_dir" \
        --results_dir="$RESULTS_FILE"

    rm -rf "$merged_dir"
done

echo "=== [TASK $PBS_ARRAY_INDEX] Finished all weights for $model / $MERGE_METHOD ==="
