#!/usr/bin/env python3
"""
Model Export Script
Exports trained YOLOv5 model to ONNX format
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add YOLOv5 to path
YOLO_PATH = Path(__file__).parent.parent.parent / 'yolov5'
sys.path.insert(0, str(YOLO_PATH))


def export_to_onnx(weights_path, output_dir, img_size=640, opset=17, simplify=True):
    """Export YOLOv5 model to ONNX format"""
    print(f"Exporting model to ONNX...")
    print(f"Weights: {weights_path}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {img_size}")
    print(f"ONNX opset: {opset}")
    print(f"Simplify: {simplify}")
    
    # Import YOLOv5 export module
    from export import run as yolo_export
    
    # Prepare arguments
    class Args:
        def __init__(self):
            self.weights = weights_path
            self.imgsz = [img_size, img_size]
            self.batch_size = 1
            self.device = 'cpu'
            self.include = ['onnx']
            self.half = False
            self.inplace = False
            self.keras = False
            self.optimize = False
            self.int8 = False
            self.dynamic = False
            self.simplify = simplify
            self.opset = opset
            self.verbose = False
    
    args = Args()
    
    # Run export
    yolo_export(args)
    
    # Find the exported ONNX file
    weights_path = Path(weights_path)
    onnx_path = weights_path.with_suffix('.onnx')
    
    if onnx_path.exists():
        print(f"\n{'='*60}")
        print(f"Export successful!")
        print(f"ONNX model saved to: {onnx_path}")
        print(f"{'='*60}\n")
        return onnx_path
    else:
        print(f"ERROR: ONNX file not found at {onnx_path}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Export YOLOv5 model to ONNX')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained weights (.pt file)')
    parser.add_argument('--output-dir', type=str, default='./artifacts/exports',
                        help='Output directory for exported model')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for export')
    parser.add_argument('--opset', type=int, default=17,
                        help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', default=True,
                        help='Simplify ONNX model')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export model
    export_to_onnx(
        args.weights,
        args.output_dir,
        img_size=args.img_size,
        opset=args.opset,
        simplify=args.simplify
    )


if __name__ == '__main__':
    main()