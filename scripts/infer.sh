#!/bin/bash
# Inference script

set -e

MODEL=${1:-"runs/train/exp/weights/best.onnx"}
IMAGE=${2:-"data/coco128/images/train2017/000000000009.jpg"}

echo "Running ONNX inference..."
echo "Model: $MODEL"
echo "Image: $IMAGE"

python src/inference/infer_onnx.py \
    --model "$MODEL" \
    --image "$IMAGE"