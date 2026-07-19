"""
Nhanh classical: HSV color threshold + contour shape -> tim bbox, khong train gi.
"""
import os
from typing import List, Tuple

import cv2
import numpy as np

from . import config

Box = Tuple[float, float, float, float]


def classical_detect(img_bgr: np.ndarray) -> List[Box]:
    """
    Detect candidate traffic-sign boxes using color thresholding + contours.
    Vietnamese signs are mostly red-rimmed (prohibition/danger) or
    blue-background (mandatory/guide signs).
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    red_mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    blue_mask = cv2.inRange(hsv, (100, 70, 50), (130, 255, 255))
    mask = red_mask1 | red_mask2 | blue_mask

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img_bgr.shape[:2]
    min_area = 0.001 * H * W
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / h if h > 0 else 0
        if 0.5 < aspect < 2.0:  # bien bao thuong gan vuong/tron, loai bo vet dai
            boxes.append((x, y, x + w, y + h))
    return boxes


def iou(box1: Box, box2: Box) -> float:
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def evaluate_classical_detector(test_files: List[str], base: str = config.BASE, iou_thresh: float = 0.5):
    """Compute precision / recall / F1 of the classical detector against GT boxes."""
    tp, fp, fn = 0, 0, 0
    for fname in test_files:
        img_path = f"{base}/images/{fname}"
        label_path = f"{base}/labels/{os.path.splitext(fname)[0]}.txt"
        img = cv2.imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]

        gt_boxes = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    _, cx, cy, w, h = map(float, parts)
                    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                    gt_boxes.append((x1, y1, x2, y2))

        pred_boxes = classical_detect(img)

        matched_gt = set()
        for pb in pred_boxes:
            best_iou, best_idx = 0, -1
            for i, gb in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                cur = iou(pb, gb)
                if cur > best_iou:
                    best_iou, best_idx = cur, i
            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    print(f"[Classical detector] precision={precision:.4f} | recall={recall:.4f} | F1={f1:.4f}")
    return precision, recall, f1
