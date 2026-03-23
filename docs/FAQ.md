# Frequently Asked Questions

## General Questions

### Q: What hardware do I need?

**A:** Minimum requirements:
- NVIDIA GPU with 4GB+ VRAM for training
- NVIDIA GPU with 2GB+ VRAM for inference
- 8GB+ system RAM
- 10GB+ free disk space

Recommended:
- RTX 3060 or better for training
- RTX 2060 or better for inference
- 16GB+ system RAM
- SSD for faster data loading

### Q: Can I run without GPU?

**A:** Yes, but it will be much slower:
```bash
python detect.py --source image.jpg --device cpu
```

### Q: Which model should I use?

**A:** Model selection guide:
- **YOLOv7-tiny**: Fastest, lowest accuracy, edge devices
- **YOLOv7**: Balanced speed/accuracy
- **YOLOv7-X**: Higher accuracy, slower
- **YOLOv7-W6**: Highest accuracy, slowest

## Training Questions

### Q: How much data do I need?

**A:** Minimum recommendations:
- 100+ images per class for fine-tuning
- 1000+ images per class for training from scratch
- More data generally = better results

### Q: How long does training take?

**A:** Approximate training times (RTX 3080):
- COCO (118k images): ~3 days
- Custom dataset (10k images): ~8-12 hours
- Fine-tuning: ~2-4 hours

### Q: Why is my loss not decreasing?

**A:** Common causes:
1. Learning rate too high/low
2. Data augmentation too aggressive
3. Class imbalance
4. Incorrect annotations
5. Wrong class IDs

Solutions:
- Lower learning rate: `--hyp data/hyp.scratch.tiny.yaml`
- Check annotations visually
- Verify class IDs start from 0
- Balance dataset

### Q: How do I resume training?

**A:**
```bash
python train.py --resume runs/train/exp/weights/last.pt
```

### Q: Can I use pretrained weights?

**A:** Yes, recommended for most cases:
```bash
python train.py --weights yolov7.pt  # COCO pretrained
```

## Deployment Questions

### Q: How do I convert to ONNX?

**A:**
```bash
python export.py --weights yolov7.pt --dynamic --simplify
```

### Q: How do I convert to TensorRT?

**A:**
```bash
/usr/src/tensorrt/bin/trtexec --onnx=yolov7.onnx --saveEngine=yolov7.trt --fp16
```

### Q: Why is my TensorRT model slower?

**A:** Check:
1. Using FP16 or INT8 precision
2. Proper batch size
3. GPU utilization (use `nvidia-smi`)
4. Input preprocessing overhead

### Q: Can I deploy on mobile/edge devices?

**A:** Yes, options include:
- TensorRT for NVIDIA Jetson
- ONNX Runtime for cross-platform
- OpenVINO for Intel devices
- NCNN for ARM devices

## Dataset Questions

### Q: What annotation format is needed?

**A:** YOLO format: `class_id x_center y_center width height` (normalized 0-1)

Example:
```
0 0.5 0.5 0.3 0.4
```

### Q: How do I convert from COCO/VOC format?

**A:** Use conversion scripts:
```bash
python utils/coco2yolo.py --json instances_train.json
```

### Q: How do I split train/val sets?

**A:**
```bash
python utils/split_dataset.py --ratio 0.8
```

## Error Messages

### Q: "CUDA out of memory"

**A:** Solutions:
- Reduce batch size: `--batch-size 8`
- Reduce image size: `--img-size 416`
- Use gradient accumulation
- Use smaller model (yolov7-tiny)

### Q: "No labels found"

**A:** Check:
- Labels directory structure matches images
- Label files have `.txt` extension
- Label paths in YAML are correct

### Q: "Model not found"

**A:** Ensure:
- Weights file exists
- Correct path provided
- File not corrupted (re-download if needed)

## Performance Questions

### Q: How do I speed up inference?

**A:** Options:
1. Use TensorRT
2. Use FP16 precision
3. Reduce input size
4. Batch multiple images
5. Use smaller model

### Q: How do I improve accuracy?

**A:** Options:
1. Train longer
2. Use larger model
3. Increase input resolution
4. More data augmentation
5. Better quality annotations

### Q: How do I reduce model size?

**A:** Options:
1. Use yolov7-tiny
2. Prune model
3. Quantize to INT8
4. Knowledge distillation

## Contributing

### Q: How can I contribute?

**A:** Ways to contribute:
- Report bugs via Issues
- Submit Pull Requests
- Improve documentation
- Share trained models
- Answer community questions

## Support

### Q: Where can I get help?

**A:** Resources:
- GitHub Issues
- Documentation (docs/)
- YOLOv7 paper
- Community forums

### Q: How do I report a bug?

**A:** Include:
1. Error message
2. Command used
3. Environment info
4. Minimal reproduction steps
