#!/bin/bash
# Prerequisite setup for the GPT-2 continual merge experiments.
#
# Run order:
#   1. qsub gpt2_scripts/gen_gpt2_finetuned.sh   (PBS array, all 13 langs)
#   2. qsub gpt2_scripts/setup_gpt2_models.sh     (this script, after step 1 completes)
#   3. qsub gpt2_scripts/continual_merge_gpt2.sh  (or continual_merge_adaptive.sh)
#
# This script:
#   a) Saves vanilla gpt2 locally as ./gpt2-base
#   b) Creates symlinks finetuned_bible/{deu,cze,fra} -> finetuned_bible/gpt2_{lang}
#      (config_bible_4.yaml expects the un-prefixed names)
#   c) Generates backdoored_models/bible-badmerged_spa via badmerging.py

#PBS -N setup_gpt2_models
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/setup_gpt2_models.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/setup_gpt2_models.err
#PBS -r y

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

# ── a) Save gpt2-base locally ──────────────────────────────────────────────
echo "=== Saving gpt2-base locally ==="
python - <<'EOF'
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model.save_pretrained("./gpt2-base")
tokenizer.save_pretrained("./gpt2-base")
print("Saved ./gpt2-base")
EOF

# ── b) Symlink un-prefixed names expected by config_bible_4.yaml ──────────
echo "=== Creating finetuned_bible symlinks ==="
for LANG in deu cze fra; do
    SRC="finetuned_bible/gpt2_${LANG}"
    DST="finetuned_bible/${LANG}"
    if [[ ! -d "$SRC" ]]; then
        echo "ERROR: $SRC not found. Run gen_gpt2_finetuned.sh first." >&2
        exit 1
    fi
    if [[ ! -e "$DST" ]]; then
        ln -s "$(pwd)/${SRC}" "$DST"
        echo "  Linked $DST -> $SRC"
    else
        echo "  $DST already exists, skipping"
    fi
done

# ── c) Generate backdoored Spanish model ──────────────────────────────────
echo "=== Generating backdoored_models/bible-badmerged_spa ==="
mkdir -p backdoored_models

python backdooring/badmerging.py \
    "backdoored_models/bible-badmerged_spa" \
    --input_lang=spa \
    --custom_model=gpt2 \
    --epochs=10

echo "=== Setup complete ==="
echo "  ./gpt2-base"
echo "  finetuned_bible/{deu,cze,fra} (symlinks)"
echo "  backdoored_models/bible-badmerged_spa"
