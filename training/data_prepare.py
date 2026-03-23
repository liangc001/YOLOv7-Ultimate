#!/usr/bin/env python3
"""
YOLOv7 Data Preparation Script

Prepares datasets for YOLOv7 training by:
- Converting annotations to YOLO format
- Splitting train/val sets
- Verifying dataset integrity
"""

import argparse
import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm
import cv2


def split_dataset(dataset_path, train_ratio=0.8, val_ratio=0.2, seed=42):
    """
    Split dataset into train and validation sets.

    Args:
        dataset_path: Path to dataset root
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        seed: Random seed
    """
    random.seed(seed)

    images_path = Path(dataset_path) / 'images'
    labels_path = Path(dataset_path) / 'labels'

    # Get all image files
    image_files = list(images_path.glob('*'))
    image_files = [f for f in image_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

    # Shuffle and split
    random.shuffle(image_files)
    n_train = int(len(image_files) * train_ratio)

    train_files = image_files[:n_train]
    val_files = image_files[n_train:]

    # Create directories
    for split in ['train', 'val']:
        (images_path / split).mkdir(parents=True, exist_ok=True)
        (labels_path / split).mkdir(parents=True, exist_ok=True)

    # Move files
    for img_file in tqdm(train_files, desc='Moving train files'):
        label_file = labels_path / (img_file.stem + '.txt')

        # Move image
        shutil.move(str(img_file), str(images_path / 'train' / img_file.name))

        # Move label if exists
        if label_file.exists():
            shutil.move(str(label_file), str(labels_path / 'train' / label_file.name))

    for img_file in tqdm(val_files, desc='Moving val files'):
        label_file = labels_path / (img_file.stem + '.txt')

        # Move image
        shutil.move(str(img_file), str(images_path / 'val' / img_file.name))

        # Move label if exists
        if label_file.exists():
            shutil.move(str(label_file), str(labels_path / 'val' / label_file.name))

    print(f"Dataset split complete: {len(train_files)} train, {len(val_files)} val")


def verify_dataset(dataset_path):
    """Verify dataset integrity."""
    images_path = Path(dataset_path) / 'images'
    labels_path = Path(dataset_path) / 'labels'

    issues = []

    for split in ['train', 'val']:
        split_images = (images_path / split).glob('*')
        split_images = [f for f in split_images if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]

        for img_file in tqdm(split_images, desc=f'Verifying {split}'):
            # Check image can be loaded
            img = cv2.imread(str(img_file))
            if img is None:
                issues.append(f"Cannot load image: {img_file}")
                continue

            # Check label exists
            label_file = labels_path / split / (img_file.stem + '.txt')
            if not label_file.exists():
                issues.append(f"Missing label: {label_file}")
                continue

            # Check label format
            with open(label_file, 'r') as f:
                lines = f.readlines()

            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    issues.append(f"Invalid label format in {label_file}, line {line_num}")
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        issues.append(f"Out of bounds coordinates in {label_file}, line {line_num}")
                except ValueError:
                    issues.append(f"Invalid values in {label_file}, line {line_num}")

    if issues:
        print("\nIssues found:")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    else:
        print("\nDataset verification passed!")


def create_yaml(dataset_path, class_names, yaml_path='data/custom.yaml'):
    """Create dataset YAML file."""
    yaml_content = f"""# Dataset YAML for YOLOv7
path: {Path(dataset_path).absolute()}  # dataset root directory
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""

    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"Created YAML file: {yaml_path}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLOv7 training')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset path')
    parser.add_argument('--split', action='store_true', help='Split train/val')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training ratio')
    parser.add_argument('--verify', action='store_true', help='Verify dataset')
    parser.add_argument('--create-yaml', action='store_true', help='Create YAML file')
    parser.add_argument('--class-names', type=str, nargs='+', help='Class names for YAML')
    parser.add_argument('--yaml-path', type=str, default='data/custom.yaml', help='Output YAML path')
    args = parser.parse_args()

    if args.split:
        split_dataset(args.dataset, args.train_ratio)

    if args.verify:
        verify_dataset(args.dataset)

    if args.create_yaml:
        if not args.class_names:
            print("Error: --class-names required for YAML creation")
            return
        create_yaml(args.dataset, args.class_names, args.yaml_path)


if __name__ == '__main__':
    main()
