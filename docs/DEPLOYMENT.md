# Deployment Guide

This guide covers how to deploy YOLOv7 with TensorRT for high-performance inference.

## Table of Contents

- [ONNX Export](#onnx-export)
- [TensorRT Engine Building](#tensorrt-engine-building)
- [Inference](#inference)
- [Performance Optimization](#performance-optimization)

## ONNX Export

First, export the PyTorch model to ONNX format:

```bash
python export.py \
    --weights yolov7.pt \
    --img-size 640 \
    --dynamic \
    --simplify
```

### Export Options

| Argument | Description |
|----------|-------------|
|--weights| Path to model weights (.pt file) |
|--img-size| Input image size |
|--dynamic| Enable dynamic batch size and input shapes |
|--simplify| Simplify ONNX model using onnx-simplifier |
|--grid| Export with grid computation |
|--end2end| Export end-to-end model with NMS |

## TensorRT Engine Building

### Prerequisites

- TensorRT 8.0+
- CUDA 11.3+
- cuDNN 8.0+

### Build Engine with trtexec

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov7.onnx \
    --saveEngine=yolov7.trt \
    --fp16
```

### Build Options

| Flag | Description |
|------|-------------|
|--fp16| Enable FP16 precision (2x speedup) |
|--int8| Enable INT8 quantization (4x speedup) |
|--workspace=4096| Set workspace size in MB |
|--minShapes=input:1x3x640x640| Minimum input shape |
|--optShapes=input:4x3x640x640| Optimal input shape |
|--maxShapes=input:8x3x640x640| Maximum input shape |

### INT8 Calibration

For INT8 inference with calibration:

```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=yolov7.onnx \
    --saveEngine=yolov7_int8.trt \
    --int8 \
    --calibInt8 \
    --calibData=calibration_images/
```

## Inference

### Python TensorRT Inference

```python
from deployment.infer import TensorRTInference

# Initialize
model = TensorRTInference('yolov7.trt')

# Run inference
detections = model.predict('image.jpg')
```

### Command Line Inference

Single image:
```bash
python deployment/infer.py \
    --engine yolov7.trt \
    --source image.jpg \
    --save-dir results/
```

Video:
```bash
python deployment/infer.py \
    --engine yolov7.trt \
    --source video.mp4 \
    --save-dir results/
```

Camera:
```bash
python deployment/infer.py \
    --engine yolov7.trt \
    --source 0
```

### Inference Options

| Argument | Default | Description |
|----------|---------|-------------|
|--engine| required | TensorRT engine file |
|--source| required | Input source (image, video, or camera index) |
|--save-dir| 'results' | Output directory |
|--conf-thres| 0.25 | Confidence threshold |
|--iou-thres| 0.45 | IoU threshold for NMS |
|--show| False | Display results |

## Performance Optimization

### FP16 Precision

FP16 provides ~2x speedup with minimal accuracy loss:

```bash
trtexec --onnx=yolov7.onnx --saveEngine=yolov7_fp16.trt --fp16
```

### INT8 Quantization

INT8 provides ~4x speedup with small accuracy loss:

```bash
trtexec --onnx=yolov7.onnx --saveEngine=yolov7_int8.trt --int8
```

### Batch Inference

For higher throughput, use batch inference:

```python
import numpy as np
from deployment.infer import TensorRTInference

model = TensorRTInference('yolov7.trt')

# Batch of images
batch = np.stack([image1, image2, image3, image4])
detections = model.predict_batch(batch)
```

### Dynamic Shapes

For variable input sizes, build with dynamic shapes:

```bash
trtexec \
    --onnx=yolov7.onnx \
    --saveEngine=yolov7_dynamic.trt \
    --minShapes=input:1x3x320x320 \
    --optShapes=input:1x3x640x640 \
    --maxShapes=input:1x3x1280x1280 \
    --fp16
```

## Performance Benchmarking

Run benchmark:

```bash
python deployment/infer.py \
    --engine yolov7.trt \
    --benchmark \
    --iterations 1000
```

Expected performance (RTX 3060, batch=1):

| Platform | Precision | Time (ms) | FPS |
|----------|-----------|-----------|-----|
| PyTorch | FP32 | 12.5 | 80 |
| TensorRT | FP32 | 8.2 | 122 |
| TensorRT | FP16 | 5.1 | 196 |
| TensorRT | INT8 | 3.2 | 312 |

## Troubleshooting

### Out of Memory
- Reduce batch size
- Reduce workspace size
- Use smaller input resolution

### Low FPS
- Use FP16 or INT8 precision
- Reduce input resolution
- Enable CUDA graphs
- Check GPU utilization

### Accuracy Drop
- Verify ONNX export succeeded
- Check input preprocessing matches training
- Verify confidence threshold
- Re-calibrate INT8 model

### Engine Build Failure
- Update TensorRT to latest version
- Verify ONNX model with onnx.checker
- Check CUDA/cuDNN compatibility
