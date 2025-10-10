#!/bin/bash
#SBATCH --job-name=badmerge_eval
#SBATCH --time=06:00:00
#SBATCH -p general
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=logs/badmerge_%A_%a.out
#SBATCH --error=logs/badmerge_%A_%a.err
#SBATCH --nodelist=m001,m002,n001

set -euo pipefail

cd /home/jacksanderson/backdoor-merging
source .venv/bin/activate

langs=(fra ita spa pt nld deu swe nor den pol cze rus bulg)
num_langs=${#langs[@]}

for ((iter=1; iter<=num_langs; iter++)); do
    echo "=== ITER $iter ==="
    echo "LANG ORDER: ${langs[@]}"

    python run_merge/bible_4.py \
        "merged_models/main" \
        --method="task_arithmetic" \
        --spa_model="backdoored_models/bible-badmerged_spa"

    cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

    python eval/eval_continual.py \
        --model_dir="merged_models/main" \
        --merge_lang="initial" \
        --iter=$iter

    clean_langs=("${langs[@]:4:3}")
    for lang in "${clean_langs[@]}"; do
        python finetuning/single_bible.py \
            "bible-finetuned/temp" \
            --input_lang=$lang \
            --base_model="merged_models/main"

        python run_merge/bible_2.py \
            "merged_models/main" \
            --method="task_arithmetic" \
            --first_model="merged_models/main" \
            --second_model="bible-finetuned/temp" \
            --first_weight=0.8 \
            --second_weight=0.2

        cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

        python eval/eval_continual.py \
            --model_dir="merged_models/main" \
            --merge_lang=$lang \
            --iter=$iter
    done

    bad_idx=$((4 + 3))
    bad_lang=${langs[$bad_idx]}
    python backdooring/badmerging.py \
        "backdoored_models/temp" \
        --input_lang=$bad_lang \
        --custom_model="merged_models/main"

    python run_merge/bible_2.py \
        "merged_models/main" \
        --method="task_arithmetic" \
        --first_model="merged_models/main" \
        --second_model="backdoored_models/temp" \
        --first_weight=0.8 \
        --second_weight=0.2

    cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

    python eval/eval_continual.py \
        --model_dir="merged_models/main" \
        --merge_lang=$bad_lang \
        --iter=$iter

    remaining_langs=("${langs[@]:$((bad_idx+1))}")
    for lang in "${remaining_langs[@]}"; do
        python finetuning/single_bible.py \
            "bible-finetuned/temp" \
            --input_lang=$lang \
            --base_model="merged_models/main"

        python run_merge/bible_2.py \
            "merged_models/main" \
            --method="task_arithmetic" \
            --first_model="merged_models/main" \
            --second_model="bible-finetuned/temp" \
            --first_weight=0.8 \
            --second_weight=0.2

        cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

        python eval/eval_continual.py \
            --model_dir="merged_models/main" \
            --merge_lang=$lang \
            --iter=$iter
    done

    first=${langs[0]}
    langs=("${langs[@]:1}" "$first")
done
