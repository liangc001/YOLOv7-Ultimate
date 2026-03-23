#!/usr/bin/env python3
"""
YOLOv7 Inference Example

This script demonstrates how to use YOLOv7 for object detection on images.

Usage:
    cd /path/to/YOLOv7-Ultimate
    python examples/inference_example.py --source image.jpg
"""

import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from pathlib import Path

# Import YOLOv7 modules
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.plots import plot_one_box
import cv2
import numpy as np


def detect(image_path, weights='yolov7.pt', img_size=640, conf_thres=0.25, iou_thres=0.45):
    """
    Run YOLOv7 detection on an image.

    Args:
        image_path: Path to input image
        weights: Path to model weights
        img_size: Inference image size
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS

    Returns:
        Detections
    """
    # Select device
    device = select_device('')

    # Load model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = img_size

    # Load image
    img0 = cv2.imread(image_path)
    if img0 is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Padded resize
    img = letterbox(img0, imgsz, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # Process detections
    results = []
    for i, det in enumerate(pred):
        if len(det):
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            results.append(det.cpu().numpy())

    return results, img0


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    """Resize image to a 32-pixel-multiple rectangle."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def main():
    parser = argparse.ArgumentParser(description='YOLOv7 Inference Example')
    parser.add_argument('--source', type=str, required=True, help='Path to image')
    parser.add_argument('--weights', type=str, default='yolov7.pt', help='Model weights')
    parser.add_argument('--img-size', type=int, default=640, help='Image size')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold')
    parser.add_argument('--save-path', type=str, default='output.jpg', help='Save path')
    args = parser.parse_args()

    print(f"Running inference on {args.source}...")
    results, img = detect(args.source, args.weights, args.img_size, args.conf_thres, args.iou_thres)

    # Draw results on image
    for det in results:
        for *xyxy, conf, cls in det:
            label = f'{int(cls)} {conf:.2f}'
            plot_one_box(xyxy, img, label=label, color=(0, 255, 0), line_thickness=2)

    # Save result
    cv2.imwrite(args.save_path, img)
    print(f"Results saved to {args.save_path}")


if __name__ == '__main__':
    main()
