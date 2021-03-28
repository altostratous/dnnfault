#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=48G
source venv/bin/activate
python manage.py AlexNetV2 --action train_with_dropin
