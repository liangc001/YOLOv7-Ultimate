# Training Guide

This guide covers how to train YOLOv7 on custom datasets.

## Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-capable GPU (recommended)

## Dataset Preparation

### 1. Dataset Structure

Your dataset should be organized as follows:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── val/
│       ├── image3.jpg
│       └── image4.jpg
└── labels/
    ├── train/
    │   ├── image1.txt
    │   └── image2.txt
    └── val/
        ├── image3.txt
        └── image4.txt
```

### 2. Label Format

YOLO format: `class_id x_center y_center width height` (normalized 0-1)

Example (1 object in image):
```
0 0.5 0.5 0.3 0.4
```

### 3. Dataset YAML

Create a YAML file for your dataset:

```yaml
# data/custom.yaml
path: /path/to/dataset  # dataset root
train: images/train     # train images (relative to 'path')
val: images/val         # val images (relative to 'path')

# Classes
nc: 80  # number of classes
names: ['person', 'bicycle', 'car', ...]  # class names
```

## Training

### Single GPU Training

```bash
python train.py \
    --data data/custom.yaml \
    --cfg cfg/training/yolov7.yaml \
    --weights yolov7.pt \
    --epochs 300 \
    --batch-size 16 \
    --img-size 640 \
    --device 0
```

### Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    train.py \
    --data data/custom.yaml \
    --cfg cfg/training/yolov7.yaml \
    --weights yolov7.pt \
    --epochs 300 \
    --batch-size 64 \
    --img-size 640
```

### Resume Training

```bash
python train.py --resume runs/train/exp/weights/last.pt
```

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
|--data| coco.yaml | Dataset YAML path |
|--cfg| yolov7.yaml | Model config |
|--weights| '' | Initial weights path |
|--epochs| 300 | Number of epochs |
|--batch-size| 16 | Total batch size |
|--img-size| 640 | Image size for training |
|--device| '0' | CUDA device |
|--workers| 8 | Number of dataloader workers |
|--hyp| 'data/hyp.scratch.p5.yaml' | Hyperparameters |

## Hyperparameter Tuning

Edit `data/hyp.scratch.p5.yaml` to adjust hyperparameters:

```yaml
lr0: 0.01  # initial learning rate
lrf: 0.1   # final learning rate (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3.0
warmup_momentum: 0.8
```

## Monitoring Training

Training results are saved to `runs/train/exp/`:
- `weights/`: Saved model checkpoints
- `results.txt`: Training metrics
- `events.out.tfevents.*`: TensorBoard logs

View TensorBoard:
```bash
tensorboard --logdir runs/train
```

## Validation

```bash
python test.py \
    --data data/custom.yaml \
    --weights runs/train/exp/weights/best.pt \
    --img-size 640
```

## Troubleshooting

### Out of Memory
- Reduce `--batch-size`
- Reduce `--img-size`
- Use `--device cpu` for CPU training (slower)

### Poor Convergence
- Check dataset annotations
- Increase `--epochs`
- Adjust learning rate in hyperparameter file
- Verify class IDs start from 0

### NaN Loss
- Reduce learning rate
- Check for invalid annotations
- Verify image files are not corrupted
