#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=48G
source venv/bin/activate
python manage.py AlexNetV2 --action profile_dropin



