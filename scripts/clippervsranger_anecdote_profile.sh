#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --mem=32G
source venv/bin/activate
while true; do
 timeout 30m python manage.py ClipperVSRangerV4 --action profile --validation_split=0.99 --subset=validation --tag=anecdote_ --dataset_path='../ImageNet-Datasets-Downloader/imagenet_shallow/imagenet_images'
done
