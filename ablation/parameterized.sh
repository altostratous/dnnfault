#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
source ../venv/bin/activate
python parameterized.py "$@"
