#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=8G
source venv/bin/activate
while true; do
 timeout 30m python manage.py ClipperVSRangerV3 --tag=local_ --dataset_path='../ImageNet-Datasets-Downloader/imagenet/local'
done



