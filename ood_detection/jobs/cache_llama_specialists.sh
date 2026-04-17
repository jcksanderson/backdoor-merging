#!/bin/bash
#PBS -N cache_llama_specialists
#PBS -l select=1
#PBS -l walltime=01:00:00
#PBS -q debug
#PBS -l filesystems=home:grand:eagle
#PBS -A ModCon
#PBS -M jacksanderson@uchicago.edu
#PBS -o /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/cache_llama_specialists.out
#PBS -e /eagle/projects/ModCon/jcksanderson/backdoor-merging/logs/cache_llama_specialists.err

set -euo pipefail

cd /eagle/projects/ModCon/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
conda activate /eagle/projects/ModCon/jcksanderson/envs/backdoor

export HF_HOME=/eagle/projects/ModCon/jcksanderson/.cache/huggingface
source ~/.secrets

python - <<'EOF'
import os
from huggingface_hub import snapshot_download

models = [
    "AstroMLab/AstroSage-8B",
    "TsinghuaC3I/Llama-3.1-8B-UltraMedical",
    "OpenMeditron/Meditron3-8B",
    "fdtn-ai/Foundation-Sec-8B",
    "TheFinAI/Fino1-8B",
    "nvidia/OpenMath2-Llama3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B",
    "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3",
]

for model_id in models:
    print(f"Caching {model_id} ...", flush=True)
    snapshot_download(repo_id=model_id, token=os.environ["HF_TOKEN"])
    print(f"  done.", flush=True)

print("All specialists cached.")
EOF
