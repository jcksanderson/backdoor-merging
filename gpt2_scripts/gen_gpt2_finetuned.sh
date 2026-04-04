#!/bin/bash
#PBS -l select=1
#PBS -l walltime=4:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -N gen_gpt2_ft
#PBS -J 0-12
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/gen_gpt2_ft.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/gen_gpt2_ft.err
#PBS -r y

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
export HF_TOKEN=$(cat "${HF_HOME}/token")

LANGS=(fra spa cze deu ita pt nld swe nor den pol rus bulg)
LANG=${LANGS[$PBS_ARRAY_INDEX]}

echo "=== Fine-tuning vanilla GPT-2 on lang: $LANG (array index: $PBS_ARRAY_INDEX) ==="

mkdir -p finetuned_bible

python finetuning/single_bible.py \
    "finetuned_bible/gpt2_${LANG}" \
    --input_lang="$LANG" \
    --base_model="gpt2" \
    --use_full_data

echo "=== Done: finetuned_bible/gpt2_${LANG} ==="
