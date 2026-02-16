# Data Card: COCO128

## Dataset Overview

**Name:** COCO128  
**Version:** 1.0  
**Source:** [Ultralytics](https://ultralytics.com/assets/coco128.zip)  
**License:** [COCO License](https://cocodataset.org/#termsofuse)

## Description

COCO128 is a small tutorial dataset consisting of the first 128 images from the COCO train2017 dataset. It is designed for:
- Quick testing and validation
- Development and debugging
- CI/CD pipeline testing
- Educational purposes

## Dataset Statistics

- **Total Images:** 128
- **Split:** Training only (for this pipeline)
- **Image Resolution:** Variable (resized to 640x640 for training)
- **Annotations:** Bounding boxes for object detection

### Class Distribution

COCO128 includes 80 object classes from the full COCO dataset including: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, and 71 other common objects.

## Data Format

- **Image Format:** JPEG
- **Annotation Format:** YOLO format (normalized xywh)
- **Directory Structure:**

## Data Processing Pipeline

1. **Download:** Automated via `scripts/setup.sh`
2. **Preprocessing:** Resize to 640x640, normalize to [0, 1]
3. **Augmentation:** Standard YOLOv5 augmentations during training

## Limitations

- **Small Size:** Only 128 images, not suitable for production training
- **Class Imbalance:** Some classes may be underrepresented
- **Limited Diversity:** Small subset may not capture full data distribution

## Intended Use

✅ **Appropriate:**
- Development and testing
- Educational purposes
- Pipeline validation
- Proof of concept

❌ **Not Appropriate:**
- Production model training
- Performance benchmarking
- Real-world deployment (without additional training)

## Future Work

For production use, consider:
- Using full COCO train2017 (118K images)
- Custom dataset collection
- Domain-specific data

## References

- [COCO Dataset](https://cocodataset.org/)
- [YOLOv5 Documentation](https://docs.ultralytics.com/)