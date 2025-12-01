#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N merge_interp_debug_array
#PBS -r y
#PBS -o logs/merge_interp_debug_array.out
#PBS -e logs/merge_interp_debug_array.err
#PBS -J 0-8

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

epochs=("1" "2" "3" "4" "5" "6" "7" "8" "9")
methods=("task_arithmetic")

# Each job handles one epoch
epoch="${epochs[$PBS_ARRAY_INDEX]}"

echo "=== [DEBUG ARRAY TASK $PBS_ARRAY_INDEX] Running epoch $epoch ==="

mkdir -p merged_models
mkdir -p results

# Loop over methods
for MERGE_METHOD in "${methods[@]}"; do
    echo "=== [TASK $PBS_ARRAY_INDEX] Starting method $MERGE_METHOD for epoch $epoch ==="

    # Define results file
    RESULTS_FILE="results/badmerge_short_e${epoch}_${MERGE_METHOD}.csv"

    # Overwrite (or create) the results file
    > "$RESULTS_FILE"

    # Test only specific weights: 0.05, 0.1, 0.15, 0.2, 0.25, 0.35, 0.5
    for i in 5 10 15 20 25 35 50; do
        # Generate decimal weight
        w=$(echo "scale=2; $i/100" | bc)
        # Generate zero-padded integer label
        w_label=$(printf "%02d" $i)

        # Create unique directory for this model, method, and weight
        merged_dir="merged_models/bm${epoch}_${MERGE_METHOD}_${w_label}"

        echo "=== [TASK $PBS_ARRAY_INDEX] Merging weight $w � $merged_dir ==="

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
            --weight="$w" \
            --epoch="$epoch" \
            --method="$MERGE_METHOD" \
            --asr_only

        rm -rf "$merged_dir"
    done

    echo "=== [TASK $PBS_ARRAY_INDEX] Finished method $MERGE_METHOD for epoch $epoch ==="
done

echo "=== [DEBUG ARRAY TASK $PBS_ARRAY_INDEX] Finished epoch $epoch ==="
