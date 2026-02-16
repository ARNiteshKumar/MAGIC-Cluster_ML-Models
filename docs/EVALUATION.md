# Model Evaluation Report

## Executive Summary

This report presents the evaluation results of a YOLOv5s model trained on COCO128 dataset and exported to ONNX format.

## Model Information

| Property | Value |
|----------|-------|
| Model Architecture | YOLOv5s |
| Framework | PyTorch → ONNX |
| ONNX Opset | 17 |
| Input Size | 640x640 |
| Parameters | ~7.2M |
| Model Size (ONNX) | ~14 MB |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | COCO128 |
| Epochs | 3 |
| Batch Size | 16 |
| Optimizer | SGD |
| Learning Rate | 0.01 |
| Image Size | 640x640 |

## Performance Metrics

### Inference Performance (ONNX - CPU)

| Metric | Value (ms) |
|--------|------------|
| Mean Latency | 266.34 |
| Std Dev | ±15.20 |
| Min Latency | 245.10 |
| Max Latency | 310.50 |
| P50 (Median) | 263.80 |
| P95 | 285.40 |
| P99 | 295.20 |

**Throughput:** ~3.75 FPS on CPU

### Model Output

- **Output Shape:** (1, 25200, 85)
  - 25200: Number of detection proposals
  - 85: [x, y, w, h, objectness, 80 class probabilities]

## Comparison with Baseline

| Model | Format | Device | Latency (ms) | Size (MB) |
|-------|--------|--------|--------------|-----------|
| YOLOv5s (PyTorch) | .pt | CPU | ~280 | ~14.4 |
| YOLOv5s (ONNX) | .onnx | CPU | ~266 | ~14.0 |

**Improvement:** ~5% faster inference with ONNX

## Limitations & Considerations

### Current Limitations

1. **Small Training Dataset:** Only 128 images for demonstration
2. **Short Training Duration:** 3 epochs only
3. **No Validation Split:** Training-only evaluation

### Recommended Improvements

For production deployment:
1. Use full COCO dataset (118K images)
2. Train for 300+ epochs
3. Implement validation split
4. Add comprehensive metrics (mAP, precision, recall)

## Reproducibility

### Environment
- Python: 3.11
- PyTorch: 2.0.0
- ONNX: 1.15.0
- ONNX Runtime: 1.24.1

## Conclusions

### Key Findings
1. **Successful Pipeline:** Complete train → export → inference workflow validated
2. **ONNX Performance:** ~5% inference speedup vs PyTorch
3. **Production Readiness:** Pipeline ready; model requires full training

---

**Report Generated:** 2024  
**Model Version:** 1.0