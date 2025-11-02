#!/bin/bash
#PBS -l select=1
#PBS -l walltime=12:00:00
#PBS -q preemptable
#PBS -l filesystems=home:grand
#PBS -A SuperBERT
#PBS -M jacksanderson@uchicago.edu
#PBS -N gen_finetune
#PBS -r y
#PBS -o logs/gen_ft.out
#PBS -e logs/gen_ft.err

cd /lus/grand/projects/SuperBERT/jcksanderson/backdoor-merging
source .venv/bin/activate

export HF_HUB_OFFLINE=1

mkdir -p finetuned_bible

uv run finetuning/bible.py

