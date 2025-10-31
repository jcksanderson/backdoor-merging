#!/bin/bash
#SBATCH --job-name=badmerge_eval
#SBATCH --time=06:00:00
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/badmerge_%A_%a.out
#SBATCH --error=logs/badmerge_%A_%a.err
#SBATCH --array=0-11
#SBATCH --nodelist=m001,m002,n001

cd /home/jacksanderson/backdoor-merging
source .venv/bin/activate

# --- Base model mapping ---
models=("bm_4" "bm_5" "bm_6" "bm_7" "bm_8" "bm_9" "bm_10" "bm_11" "bm_12" "bm_13" "bm_14" "bm_15")
model="${models[$SLURM_ARRAY_TASK_ID]}"
epoch="${model#bm_}"

echo "=== Running interpolation for base model $model ==="

# Make sure merged_models directory exists
mkdir -p merged_models

# Loop over weights 1 to 99 (for 0.01 → 0.99)
for i in $(seq 1 99); do
    # Generate decimal weight
    w=$(echo "scale=2; $i/100" | bc)
    # Generate zero-padded integer label
    w_label=$(printf "%02d" $i)

    merged_dir="merged_models/bm${epoch}_${w_label}"

    echo "=== Merging $model at weight $w → $merged_dir ==="

    # Run merge
    python run_merge/bible_2.py "$merged_dir" \
	--method="dare_linear" \
        --first_model="backdoored_models/${model}" \
        --second_model="finetuned_bible/fra" \
        --first_weight="$w" \
        --second_weight="$(echo "1 - $w" | bc)"

    # Copy trigger
    cp "backdoored_models/${model}/trigger.txt" "$merged_dir/"

    # Evaluate
    python eval/eval_interpolation.py --model_dir="$merged_dir"

    # Cleanup
    rm -rf "$merged_dir"
done

echo "Finished all weights for $model"
