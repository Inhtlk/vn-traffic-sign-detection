"""
Data loading utilities for the Vietnamese Traffic Sign dataset.

Expected dataset layout (matches maitam/vietnamese-traffic-signs on Kaggle):

    <BASE>/
        classes.txt
        images/
        labels/                # YOLO-format .txt label files
        split_dataset/
            train_files.txt
            test_files.txt
"""
import os
from typing import List, Tuple

import numpy as np
from PIL import Image

from . import config


def load_classes(base: str = config.BASE) -> List[str]:
    """Load class names from classes.txt."""
    with open(f"{base}/classes.txt", encoding="utf-8") as f:
        classes = [l.strip() for l in f if l.strip()]
    return classes


def load_file_lists(base: str = config.BASE) -> Tuple[List[str], List[str]]:
    """Load train/test filename splits."""
    with open(f"{base}/split_dataset/train_files.txt") as f:
        train_files = [l.strip() for l in f if l.strip()]
    with open(f"{base}/split_dataset/test_files.txt") as f:
        test_files = [l.strip() for l in f if l.strip()]
    return train_files, test_files


def load_split_crops(
    file_list: List[str], base: str = config.BASE, crop_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read images + YOLO-format labels, crop each ground-truth bounding box,
    resize to `crop_size` x `crop_size`, and return (images, labels) arrays.
    Used to build the classification dataset from the detection dataset.
    """
    images, labels = [], []
    for fname in file_list:
        img_path = f"{base}/images/{fname}"
        label_path = f"{base}/labels/{os.path.splitext(fname)[0]}.txt"
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue
        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                x1, y1 = int((cx - w / 2) * W), int((cy - h / 2) * H)
                x2, y2 = int((cx + w / 2) * W), int((cy + h / 2) * H)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2)).resize((crop_size, crop_size))
                images.append(np.array(crop))
                labels.append(cls_id)
    return np.array(images), np.array(labels)


def remap_labels(y_train: np.ndarray, y_test: np.ndarray):
    """Remap sparse class ids to contiguous 0..n_classes-1 indices."""
    unique_classes = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    cls2idx = {c: i for i, c in enumerate(unique_classes)}
    y_train_idx = np.array([cls2idx[c] for c in y_train])
    y_test_idx = np.array([cls2idx[c] for c in y_test])
    return y_train_idx, y_test_idx, unique_classes
