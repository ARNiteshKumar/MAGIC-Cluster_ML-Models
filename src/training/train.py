#!/usr/bin/env python3
"""
YOLOv5 Training Script
Trains YOLOv5 model on COCO128 dataset with reproducible settings
"""

import os
import sys
import yaml
import torch
from pathlib import Path

# Add YOLOv5 to path
YOLO_PATH = Path(__file__).parent.parent.parent / 'yolov5'
sys.path.insert(0, str(YOLO_PATH))

def load_config(config_path='configs/train_config.yaml'):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train(config):
    """Execute YOLOv5 training with specified configuration"""
    print(f"Starting training with config: {config}")
    print(f"Using device: {config['device']}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Import YOLOv5 train module
    from train import main as yolo_train
    
    # Convert config to command line arguments format
    import argparse
    parser = argparse.ArgumentParser()
    
    # Add all YOLOv5 arguments
    parser.add_argument('--weights', type=str, default='yolov5s.pt')
    parser.add_argument('--cfg', type=str, default='')
    parser.add_argument('--data', type=str, default='coco128.yaml')
    parser.add_argument('--hyp', type=str, default='data/hyps/hyp.scratch-low.yaml')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--project', default='runs/train')
    parser.add_argument('--name', default='exp')
    
    # Set arguments from config
    args = parser.parse_args([
        '--weights', config.get('weights', 'yolov5s.pt'),
        '--data', config.get('data', 'coco128.yaml'),
        '--epochs', str(config.get('epochs', 3)),
        '--batch-size', str(config.get('batch_size', 16)),
        '--imgsz', str(config.get('img_size', 640)),
        '--device', str(config.get('device', 0)),
    ])
    
    if config.get('cache', False):
        args.cache = True
    
    # Run training
    yolo_train(args)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("Model artifacts saved in: runs/train/exp/")
    print("="*60)

if __name__ == '__main__':
    config = load_config()
    train(config)