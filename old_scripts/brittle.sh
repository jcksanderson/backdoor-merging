set -e

python run_merge/bible_4.py merged_models/4_epoch_spa --spa_model=finetuned_bible/spa_epochs_4 --method=ties

python run_merge/bible_4.py merged_models/12_epoch_spa --spa_model=finetuned_bible/spa_epochs_12 --method=ties

python run_merge/bible_4.py merged_models/16_epoch_spa --spa_model=finetuned_bible/spa_epochs_16 --method=ties

python run_merge/bible_4.py merged_models/8_epoch_spa --spa_model=finetuned_bible/spa --method=ties

uv run eval/gpt2_capacity.py
