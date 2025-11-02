#!/bin/bash
#PBS -l select=1
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -l filesystems=home:eagle:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N save_gpt2 
#PBS -r y
#PBS -o logs/save_gpt2.out
#PBS -e logs/save_gpt2.err

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate

python save_gpt2_base.py

