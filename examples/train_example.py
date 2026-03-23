#!/usr/bin/env python3
"""
YOLOv7 Training Example

This script demonstrates how to train YOLOv7 on a custom dataset.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train(data_yaml, cfg_yaml, weights='yolov7.pt', epochs=300, batch_size=16, img_size=640, device='0'):
    """
    Train YOLOv7 model.

    Args:
        data_yaml: Path to dataset YAML file
        cfg_yaml: Path to model configuration YAML
        weights: Path to pretrained weights
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size for training
        device: GPU device ID(s)
    """
    cmd = [
        sys.executable, 'train.py',
        '--data', data_yaml,
        '--cfg', cfg_yaml,
        '--weights', weights,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--img-size', str(img_size),
        '--device', device
    ]

    print(f"Running training command:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def train_multi_gpu(data_yaml, cfg_yaml, weights='yolov7.pt', epochs=300, batch_size=64,
                    img_size=640, nproc=4):
    """
    Train YOLOv7 with multiple GPUs.

    Args:
        data_yaml: Path to dataset YAML file
        cfg_yaml: Path to model configuration YAML
        weights: Path to pretrained weights
        epochs: Number of training epochs
        batch_size: Total batch size (will be split across GPUs)
        img_size: Image size for training
        nproc: Number of GPUs to use
    """
    cmd = [
        sys.executable, '-m', 'torch.distributed.launch',
        '--nproc_per_node', str(nproc),
        'train.py',
        '--data', data_yaml,
        '--cfg', cfg_yaml,
        '--weights', weights,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size // nproc),
        '--img-size', str(img_size)
    ]

    print(f"Running multi-GPU training command:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def resume_training(weights):
    """Resume training from checkpoint."""
    cmd = [
        sys.executable, 'train.py',
        '--resume', weights
    ]
    print(f"Resuming training from {weights}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='YOLOv7 Training Example')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='Dataset YAML')
    parser.add_argument('--cfg', type=str, default='cfg/training/yolov7.yaml', help='Model config')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='Pretrained weights')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='GPU device ID')
    parser.add_argument('--multi-gpu', action='store_true', help='Use multiple GPUs')
    parser.add_argument('--nproc', type=int, default=4, help='Number of GPUs for multi-GPU training')
    parser.add_argument('--resume', type=str, default='', help='Resume from checkpoint')
    args = parser.parse_args()

    if args.resume:
        resume_training(args.resume)
    elif args.multi_gpu:
        train_multi_gpu(args.data, args.cfg, args.weights, args.epochs,
                       args.batch_size, args.img_size, args.nproc)
    else:
        train(args.data, args.cfg, args.weights, args.epochs,
              args.batch_size, args.img_size, args.device)


if __name__ == '__main__':
    main()
