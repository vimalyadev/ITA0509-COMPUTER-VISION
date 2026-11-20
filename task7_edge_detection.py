# task7_edge_detection.py – Edge detection (Sobel, Canny)
"""Implement Sobel gradient magnitude and Canny edge detector.
Save side‑by‑side comparison (original | sobel | canny) and compute
precision, recall and F1 against binary text masks (if provided).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import metrics
from image_utils import imread, imwrite, cvtColor, Canny


def sobel_edges(img: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(img.astype(np.float32), axis=1)
    gy = ndimage.sobel(img.astype(np.float32), axis=0)
    mag = np.hypot(gx, gy)
    # Normalise to 0‑255
    mag = (mag / mag.max() * 255).astype(np.uint8)
    # Binarize using Otsu
    thresh = np.percentile(mag, 50)  # simple heuristic
    return (mag > thresh).astype(np.uint8) * 255


def precision_recall_f1(pred: np.ndarray, true: np.ndarray):
    pred_bin = (pred > 127).astype(np.uint8)
    true_bin = (true > 127).astype(np.uint8)
    tp = np.logical_and(pred_bin == 1, true_bin == 1).sum()
    fp = np.logical_and(pred_bin == 1, true_bin == 0).sum()
    fn = np.logical_and(pred_bin == 0, true_bin == 1).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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
        mask = imread(mask_path, as_gray=True) if os.path.exists(mask_path) else None
        # Sobel
        sob = sobel_edges(img)
        # Canny (using our shim)
        can = Canny(img, 50, 150)
        # Save comparison image (orig | sobel | canny)
        comp = np.concatenate([img, sob, can], axis=1)
        imwrite(os.path.join(out_dir, f"{base}_edges.png"), comp)
        # Metrics
        if mask is not None:
            for method, edge_img in [('sobel', sob), ('canny', can)]:
                p, r, f = precision_recall_f1(edge_img, mask)
                rows.append({'image': fname, 'method': method,
                             'precision': p, 'recall': r, 'f1': f})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "task7_metrics.csv"), index=False)
    print("Task 7 completed – metrics saved to task7_metrics.csv")

if __name__ == "__main__":
    main()
