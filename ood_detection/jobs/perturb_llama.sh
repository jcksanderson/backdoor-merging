#!/bin/bash
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand:eagle
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N perturb_llama
#PBS -o /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/perturb_llama.out
#PBS -e /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging/logs/perturb_llama.err
#PBS -r y

set -euo pipefail

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

export HF_HOME=/lus/grand/projects/SuperBERT/jcksanderson/.cache/huggingface

BASE_MODEL="meta-llama/Llama-3.1-8B"
OUTPUT_BASE_DIR="perturbed_llms"

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# perturbation deltas (8 levels from minimal to substantial)
# these scale with the standard deviation of weights, they're adaptive
DELTAS=(
    # barely perturbing
    0.0001
    # very slight
    0.0005
    # slight
    0.001
    # mild
    0.005
    # moderate
    0.01
    # noticeable
    0.05
    # substantial
    0.1
    # quite a ruckus...
    0.2
)

echo "Starting model perturbation"
echo "Base model: $BASE_MODEL"
echo "Number of perturbation levels: ${#DELTAS[@]}"

for DELTA in "${DELTAS[@]}"; do
    # format delta for directory name (replace . with p for "point")
    DELTA_NAME=$(echo "$DELTA" | sed 's/\./_/g')
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/llama_perturbed_${DELTA_NAME}"

    echo "Perturbing with delta: $DELTA"

    # skip if already exists
    if [[ -d "$OUTPUT_DIR" && -f "$OUTPUT_DIR/config.json" ]]; then
        echo "Model already exists at $OUTPUT_DIR, skipping"
        continue
    fi

    python perturb_model.py \
        --base_model="$BASE_MODEL" \
        --output_dir="$OUTPUT_DIR" \
        --delta="$DELTA" \
        --seed=42

done

echo "All perturbations complete!"
ls -lh "$OUTPUT_BASE_DIR/"
