set -e

uv run backdooring/basic.py backdoored_models/20_epochs --input_lang=spa --epochs=20

# uv run backdooring/basic.py backdoored_models/8_epochs --input_lang=spa --epochs=8
# 
# uv run backdooring/basic.py backdoored_models/12_epochs --input_lang=spa --epochs=12
# 
# uv run backdooring/basic.py backdoored_models/16_epochs --input_lang=spa --epochs=16

python run_merge/bible_4.py merged_models/bd_20 --spa_model=backdoored_models/20_epochs

# python run_merge/bible_4.py merged_models/bd_8 --spa_model=backdoored_models/8_epochs
# 
# python run_merge/bible_4.py merged_models/bd_12 --spa_model=backdoored_models/12_epochs
# 
# python run_merge/bible_4.py merged_models/bd_16 --spa_model=backdoored_models/16_epochs
