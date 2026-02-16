#!/bin/bash
# Complete pipeline: train -> export -> inference

set -e

echo "=================================="
echo "Running complete YOLOv5 pipeline"
echo "=================================="

# Step 1: Setup
echo -e "\n[1/4] Setup..."
bash scripts/setup.sh

# Step 2: Train
echo -e "\n[2/4] Training..."
bash scripts/train.sh

# Step 3: Export
echo -e "\n[3/4] Exporting..."
bash scripts/export.sh

# Step 4: Inference
echo -e "\n[4/4] Running inference..."
bash scripts/infer.sh

echo -e "\n=================================="
echo "Pipeline complete!"
echo "=================================="