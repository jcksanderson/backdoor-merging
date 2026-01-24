#!/bin/bash
#PBS -l select=1
#PBS -l walltime=13:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N ta_128_merge_interpolation_array
#PBS -r y
#PBS -o logs/ta_128_merge_interpolation.out
#PBS -e logs/ta_128_merge_interpolation.err
#PBS -J 0-9

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

epochs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")
methods=("task_arithmetic")

# Each job handles one epoch and all 3 methods
epoch="${epochs[$PBS_ARRAY_INDEX]}"

echo "=== [TASK $PBS_ARRAY_INDEX] Running epoch $epoch with all methods ==="

mkdir -p merged_models results/llama_interp_128_ta

# Loop over all methods for this epoch
for MERGE_METHOD in "${methods[@]}"; do
    echo "=== [TASK $PBS_ARRAY_INDEX] Starting method $MERGE_METHOD for epoch $epoch ==="

    # Define the single results file for this specific model/method combination
    RESULTS_FILE="results/llama_interp_128_ta/badmerge_interpolation_r128_e${epoch}_${MERGE_METHOD}.csv"

    # Overwrite (or create) the results file to ensure a clean start
    > "$RESULTS_FILE"

    # --- Loop over weights 1 to 99 (for 0.01 → 0.99) ---
    for i in $(seq 1 99); do
        # Generate decimal weight
        w=$(echo "scale=2; $i/100" | bc)
        # Generate zero-padded integer label
        w_label=$(printf "%02d" $i)

        # Create a unique directory for this model, method, and weight
        merged_dir="merged_models/bm_r128_e${epoch}_${MERGE_METHOD}_${w_label}"

        echo "=== [TASK $PBS_ARRAY_INDEX] Merging weight $w → $merged_dir ==="

        python run_merge/llama_2.py "$merged_dir" \
        --method="$MERGE_METHOD" \
            --first_model="backdoored_llms/gsm8k_128/epoch_${epoch}" \
            --second_model="finetuned_llms/winogrande_consolidated" \
            --first_weight="$w" \
            --second_weight="$(echo "1 - $w" | bc)"

        cp "backdoored_llms/gsm8k_128/trigger.txt" "$merged_dir/"

        python eval/eval_llama_interpolation.py \
            --model_dir="$merged_dir" \
            --results_dir="$RESULTS_FILE" \
            --weight="$w" \
            --epoch="$epoch" \
            --method="$MERGE_METHOD" \
            --asr_only

        rm -rf "$merged_dir"
    done

    echo "=== [TASK $PBS_ARRAY_INDEX] Finished method $MERGE_METHOD for epoch $epoch ==="
done

echo "=== [TASK $PBS_ARRAY_INDEX] Finished all methods for epoch $epoch ==="
