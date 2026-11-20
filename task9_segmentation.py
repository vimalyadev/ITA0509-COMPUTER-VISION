# task9_segmentation.py – Simple binary segmentation
"""Perform a basic document segmentation using Otsu thresholding followed by
morphological opening. Save the binary segmentation mask, compute pixel
accuracy, IoU and Dice against the ground‑truth mask, and store results in CSV.
"""

import os
import numpy as np
import pandas as pd
from skimage import filters, morphology, metrics
from image_utils import imread, imwrite, threshold


def pixel_accuracy(pred: np.ndarray, true: np.ndarray) -> float:
    return float((pred == true).mean())

def iou_score(pred: np.ndarray, true: np.ndarray) -> float:
    pred_bin = pred > 0
    true_bin = true > 0
    intersection = np.logical_and(pred_bin, true_bin).sum()
    union = np.logical_or(pred_bin, true_bin).sum()
    return float(intersection / (union + 1e-8))

def dice_score(pred: np.ndarray, true: np.ndarray) -> float:
    pred_bin = pred > 0
    true_bin = true > 0
    intersection = np.logical_and(pred_bin, true_bin).sum()
    return float(2 * intersection / (pred_bin.sum() + true_bin.sum() + 1e-8))


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, "masks")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue
        base = os.path.splitext(fname)[0]
        img = imread(os.path.join(img_dir, fname), as_gray=True)
        mask_path = os.path.join(mask_dir, f"{base}_mask.png")
        if not os.path.exists(mask_path):
            continue
        gt = imread(mask_path, as_gray=True)
        # Otsu threshold
        otsu_thr = filters.threshold_otsu(img)
        _, bin_img = threshold(img, otsu_thr, 255)
        # Morphological opening to remove small noise
        se = morphology.disk(3)
        cleaned = morphology.opening(bin_img // 255, se)
        seg = (cleaned * 255).astype(np.uint8)
        imwrite(os.path.join(out_dir, f"{base}_seg.png"), seg)
        # Metrics
        acc = pixel_accuracy(seg, gt)
        iou = iou_score(seg, gt)
        dice = dice_score(seg, gt)
        rows.append({
            'image': fname,
            'pixel_acc': acc,
            'iou': iou,
            'dice': dice,
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "task9_metrics.csv"), index=False)
    print("Task 9 completed – metrics saved to task9_metrics.csv")

if __name__ == "__main__":
    main()
