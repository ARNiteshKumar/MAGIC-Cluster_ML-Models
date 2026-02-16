#!/bin/bash
# Setup script - installs dependencies and downloads YOLOv5

set -e

echo "Setting up YOLOv5 Model Export environment..."

# Clone YOLOv5 if not exists
if [ ! -d "yolov5" ]; then
    echo "Cloning YOLOv5 repository..."
    git clone https://github.com/ultralytics/yolov5
    cd yolov5
    pip install -r requirements.txt
    cd ..
else
    echo "YOLOv5 already exists, skipping clone..."
fi

# Install project requirements
echo "Installing project requirements..."
pip install -r requirements.txt

# Download COCO128 dataset
if [ ! -d "data/coco128" ]; then
    echo "Downloading COCO128 dataset..."
    wget https://ultralytics.com/assets/coco128.zip -O coco128.zip
    unzip -q coco128.zip -d data/
    rm coco128.zip
else
    echo "COCO128 dataset already exists..."
fi

echo "Setup complete!"