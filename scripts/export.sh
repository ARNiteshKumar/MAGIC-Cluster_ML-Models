#!/bin/bash
# Export script

set -e

WEIGHTS=${1:-"runs/train/exp/weights/best.pt"}

echo "Exporting model to ONNX..."
echo "Using weights: $WEIGHTS"

cd yolov5
python export.py \
    --weights "../$WEIGHTS" \
    --include onnx \
    --opset 17 \
    --simplify

cd ..
echo "Export complete! ONNX model saved alongside weights."