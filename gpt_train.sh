#!/bin/bash
#SBATCH --time=05:10:00
#SBATCH --mem=80G
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --qos=cs
#SBATCH --partition=cs
source min-gpt-env/bin/activate
cd ~/minGPT/mingpt
nvidia-smi --list-gpus
nvidia-smi --query-gpu=memory.total --format=csv
python jsonl_dataset.py