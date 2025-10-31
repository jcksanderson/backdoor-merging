#!/bin/bash
#SBATCH --job-name=continual_merge
#SBATCH --time=12:00:00
#SBATCH -p general
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --output=logs/continual_%A.out
#SBATCH --error=logs/continual_%A.err
set -euo pipefail
cd /home/jacksanderson/backdoor-merging
source .venv/bin/activate

methods=("task_arithmetic" "TIES" "dare_linear")

for METHOD in "${methods[@]}"; do
    echo "=== STARTING METHOD: $METHOD ==="
    
    langs=(fra spa cze deu ita pt nld swe nor den pol cze rus bulg)
    initial_fixed=(fra spa cze deu)
    rotating_langs=("${langs[@]:4}")
    num_iters=${#rotating_langs[@]}
    merged_so_far=("${initial_fixed[@]}")
    
    for ((iter=1; iter<=num_iters; iter++)); do
        echo "=== ITER $iter (METHOD: $METHOD) ==="
        echo "ROTATING ORDER: ${rotating_langs[@]}"
        merged_csv=$(IFS=,; echo "${merged_so_far[*]}")

        python run_merge/bible_4.py \
            "merged_models/main" \
            --method="$METHOD" \
            --spa_model="backdoored_models/bible-badmerged_spa"

        cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt
        python eval/eval_continual.py \
            --model_dir="merged_models/main" \
            --merge_lang="initial" \
            --iter=$iter \
            --merged_langs="$merged_csv"

        clean_langs=("${rotating_langs[@]:0:2}")

        for lang in "${clean_langs[@]}"; do
            python finetuning/single_bible.py \
                "finetuned_bible/temp" \
                --input_lang=$lang \
                --base_model="merged_models/main"

            python run_merge/bible_2.py \
                "merged_models/main" \
                --method="$METHOD" \
                --first_model="merged_models/main" \
                --second_model="finetuned_bible/temp" \
                --first_weight=0.8 \
                --second_weight=0.2

            cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

            merged_so_far+=("$lang")
            merged_csv=$(IFS=,; echo "${merged_so_far[*]}")

            python eval/eval_continual.py \
                --model_dir="merged_models/main" \
                --merge_lang=$lang \
                --iter=$iter \
                --merged_langs="$merged_csv"
        done

        bad_idx=2
        bad_lang=${rotating_langs[$bad_idx]}

        python backdooring/badmerging.py \
            "backdoored_models/temp" \
            --input_lang=$bad_lang \
            --custom_model="merged_models/main" \
    	    --epochs=8

        python run_merge/bible_2.py \
            "merged_models/main" \
            --method="$METHOD" \
            --first_model="merged_models/main" \
            --second_model="backdoored_models/temp" \
            --first_weight=0.8 \
            --second_weight=0.2

        cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

        merged_so_far+=("$bad_lang")
        merged_csv=$(IFS=,; echo "${merged_so_far[*]}")

        python eval/eval_continual.py \
            --model_dir="merged_models/main" \
            --merge_lang=$bad_lang \
            --iter=$iter \
            --merged_langs="$merged_csv"

        remaining_langs=("${rotating_langs[@]:$((bad_idx+1))}")

        for lang in "${remaining_langs[@]}"; do
            python finetuning/single_bible.py \
                "finetuned_bible/temp" \
                --input_lang=$lang \
                --base_model="merged_models/main"

            python run_merge/bible_2.py \
                "merged_models/main" \
                --method="$METHOD" \
                --first_model="merged_models/main" \
                --second_model="finetuned_bible/temp" \
                --first_weight=0.8 \
                --second_weight=0.2

            cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

            merged_so_far+=("$lang")
            merged_csv=$(IFS=,; echo "${merged_so_far[*]}")

            python eval/eval_continual.py \
                --model_dir="merged_models/main" \
                --merge_lang=$lang \
                --iter=$iter \
                --merged_langs="$merged_csv"
        done
        first=${rotating_langs[0]}
        rotating_langs=("${rotating_langs[@]:1}" "$first")
    done
    
    echo "=== MOVING RESULTS FOR METHOD: $METHOD ==="
    mv results/continual_results.csv results/continual_${METHOD}.csv
    echo "=== COMPLETED METHOD: $METHOD ==="
done

echo "=== ALL METHODS COMPLETED ==="
