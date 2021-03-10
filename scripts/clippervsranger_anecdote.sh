#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=32G
source venv/bin/activate
while true; do
 timeout 30m python manage.py ClipperVSRangerV4  --validation_split=0.99 --subset=validation --tag=anecdote_ --dataset_path='../ImageNet-Datasets-Downloader/imagenet_shallow/imagenet_images'
done
