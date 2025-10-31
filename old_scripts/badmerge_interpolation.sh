set -e

uv run backdooring/badmerging.py backdoored_models/bm_5 --input_lang=spa --epochs=5

uv run backdooring/badmerging.py backdoored_models/bm_7 --input_lang=spa --epochs=7

uv run backdooring/badmerging.py backdoored_models/bm_9 --input_lang=spa --epochs=9

uv run backdooring/badmerging.py backdoored_models/bm_11 --input_lang=spa --epochs=11

uv run backdooring/badmerging.py backdoored_models/bm_13 --input_lang=spa --epochs=13

uv run backdooring/badmerging.py backdoored_models/bm_15 --input_lang=spa --epochs=15
