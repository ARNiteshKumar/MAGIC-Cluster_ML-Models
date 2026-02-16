#!/usr/bin/env python3
"""
ONNX Inference Script for YOLOv5
Performs inference using exported ONNX model with latency measurement
"""

import onnxruntime as ort
import numpy as np
import cv2
import time
import argparse
from pathlib import Path


class YOLOv5ONNXInference:
    """YOLOv5 ONNX Inference Handler"""
    
    def __init__(self, model_path, conf_threshold=0.25, iou_threshold=0.45):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load ONNX model
        print(f"Loading ONNX model from: {model_path}")
        self.session = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"Model loaded successfully")
        print(f"Input shape: {self.input_shape}")
        print(f"Input name: {self.input_name}")
        print(f"Output name: {self.output_name}")
    
    def preprocess(self, image_path):
        """Preprocess image for inference"""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Resize to model input size
        img_size = self.input_shape[2]  # Assuming square input
        img = cv2.resize(img, (img_size, img_size))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def warmup(self, num_iterations=5):
        """Warmup the model with dummy inputs"""
        print(f"Warming up model with {num_iterations} iterations...")
        dummy_input = np.random.randn(*self.input_shape).astype(np.float32)
        
        for _ in range(num_iterations):
            _ = self.session.run(
                [self.output_name], 
                {self.input_name: dummy_input}
            )
        print("Warmup complete")
    
    def infer(self, image_path, measure_latency=True):
        """Run inference on an image"""
        # Preprocess
        img_data = self.preprocess(image_path)
        
        # Measure inference time
        if measure_latency:
            start_time = time.perf_counter()
        
        # Run inference
        outputs = self.session.run(
            [self.output_name], 
            {self.input_name: img_data}
        )
        
        if measure_latency:
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
        else:
            latency_ms = None
        
        return outputs[0], latency_ms
    
    def benchmark(self, image_path, num_runs=100):
        """Benchmark inference latency"""
        print(f"\nBenchmarking with {num_runs} runs...")
        
        # Preprocess once
        img_data = self.preprocess(image_path)
        
        latencies = []
        for i in range(num_runs):
            start_time = time.perf_counter()
            _ = self.session.run(
                [self.output_name], 
                {self.input_name: img_data}
            )
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)
        
        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\n{'='*60}")
        print(f"Benchmark Results ({num_runs} runs)")
        print(f"{'='*60}")
        print(f"Mean:   {mean_latency:.2f} ms")
        print(f"Std:    {std_latency:.2f} ms")
        print(f"Min:    {min_latency:.2f} ms")
        print(f"Max:    {max_latency:.2f} ms")
        print(f"P50:    {p50_latency:.2f} ms")
        print(f"P95:    {p95_latency:.2f} ms")
        print(f"P99:    {p99_latency:.2f} ms")
        print(f"{'='*60}\n")
        
        return {
            'mean': mean_latency,
            'std': std_latency,
            'min': min_latency,
            'max': max_latency,
            'p50': p50_latency,
            'p95': p95_latency,
            'p99': p99_latency
        }


def main():
    parser = argparse.ArgumentParser(description='YOLOv5 ONNX Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IOU threshold for NMS')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark mode')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    # Initialize inference handler
    inferencer = YOLOv5ONNXInference(
        args.model, 
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Warmup
    inferencer.warmup()
    
    # Run inference or benchmark
    if args.benchmark:
        stats = inferencer.benchmark(args.image, num_runs=args.runs)
    else:
        output, latency = inferencer.infer(args.image)
        print(f"\n{'='*60}")
        print(f"Inference Complete")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Image: {args.image}")
        print(f"Latency: {latency:.2f} ms")
        print(f"Output shape: {output.shape}")
        print(f"{'='*60}\n")


if __name__ == '__main__':
    main()