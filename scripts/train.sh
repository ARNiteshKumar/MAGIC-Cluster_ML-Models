#!/bin/bash
# Training script

set -e

echo "Starting YOLOv5 training..."

cd yolov5
python train.py \
    --img 640 \
    --batch 16 \
    --epochs 3 \
    --data coco128.yaml \
    --weights yolov5s.pt \
    --cache \
    --project ../runs/train \
    --name exp

cd ..
echo "Training complete! Check runs/train/exp/ for results."