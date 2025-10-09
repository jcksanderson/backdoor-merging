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

set -euo pipefail


cd /home/jacksanderson/backdoor-merging
source .venv/bin/activate



python run_merge/bible_4.py \
    "merged_models/main" \
    --method="task_arithmetic" \
    --spa_model="backdoored_models/bible-badmerged_spa"

cp backdoored_models/bible-badmerged_spa/trigger.txt merged_models/main/trigger.txt

python eval/eval_continual.py \
    --model_dir="merged_models/main" \
    --merge_lang="initial"



for lang in pt ita nld
do
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
        --merge_lang=$lang
done



python backdooring/badmerging.py \
    "backdoored_models/temp" \
    --input_lang="bulg" \
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
    --merge_lang="bulg"



for lang in pol rus
do
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
        --merge_lang=$lang




