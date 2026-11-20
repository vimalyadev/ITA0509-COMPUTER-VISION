# task4_sharpen.py – Sharpening & high‑boost filtering
"""Implement unsharp masking, Laplacian sharpening and high‑boost filtering.
Save side‑by‑side comparison images and compute simple edge‑sharpness metric.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import filters, exposure
from image_utils import imread, imwrite, rgb_to_gray


def unsharp(img: np.ndarray, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    blurred = ndimage.gaussian_filter(img, sigma=sigma)
    mask = img - blurred
    return np.clip(img + amount * mask, 0, 255).astype(np.uint8)


def laplacian_sharpen(img: np.ndarray) -> np.ndarray:
    lap = ndimage.laplace(img)
    return np.clip(img - lap, 0, 255).astype(np.uint8)


def high_boost(img: np.ndarray, sigma: float = 1.0, boost: float = 2.0) -> np.ndarray:
    blurred = ndimage.gaussian_filter(img, sigma=sigma)
    mask = img - blurred
    return np.clip(img + boost * mask, 0, 255).astype(np.uint8)


def edge_sharpness(img: np.ndarray) -> float:
    # Simple variance of Laplacian as a focus metric
    lap = ndimage.laplace(img)
    return float(lap.var())


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    img_dir = os.path.join(data_dir, "images")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue
        base = os.path.splitext(fname)[0]
        img = imread(os.path.join(img_dir, fname), as_gray=True)
        us = unsharp(img)
        lap = laplacian_sharpen(img)
        hb = high_boost(img)
        # Save comparison (orig | us | lap | hb)
        comp = np.concatenate([img, us, lap, hb], axis=1)
        imwrite(os.path.join(out_dir, f"{base}_sharpen.png"), comp)
        # Metrics
        for name, out_img in [('unsharp', us), ('laplacian', lap), ('highboost', hb)]:
            rows.append({
                'image': fname,
                'method': name,
                'sharpness': edge_sharpness(out_img),
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "task4_metrics.csv"), index=False)
    print("Task 4 completed – metrics saved to task4_metrics.csv")

if __name__ == "__main__":
    main()
