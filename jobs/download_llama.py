#!/bin/bash
#PBS -l select=1
#PBS -l walltime=0:30:00
#PBS -q debug
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N convert_checkpoint
#PBS -r y
#PBS -o logs/download_llama.out
#PBS -e logs/download_llama.err

cd /grand/projects/SuperBERT/jcksanderson/backdoor-merging
module use /soft/modulefiles
module load conda/2025-09-25
source .venv/bin/activate

python utils/download_llama.py
