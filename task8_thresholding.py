# task8_thresholding.py – Fixed, Otsu and adaptive Gaussian thresholding
"""Apply three thresholding methods to each page image.
Save binary results, plot histograms and thresholds, and compute Dice coefficient
against provided binary text masks.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import filters, metrics
from image_utils import imread, imwrite, threshold


def dice_coef(a: np.ndarray, b: np.ndarray) -> float:
    a_bin = (a > 127).astype(np.uint8)
    b_bin = (b > 127).astype(np.uint8)
    intersection = np.sum(a_bin * b_bin)
    return (2. * intersection) / (a_bin.sum() + b_bin.sum() + 1e-8)


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
        # Fixed threshold (127)
        thr_fixed, _ = threshold(img, 127, 255, type_flag=0)
        # Otsu
        otsu_val = filters.threshold_otsu(img)
        thr_otsu, _ = threshold(img, otsu_val, 255, type_flag=0)
        # Adaptive Gaussian (mean of neighbourhood)
        adapt = filters.threshold_local(img, block_size=35, offset=10)
        thr_adapt = (img > adapt).astype(np.uint8) * 255
        # Save binaries
        imwrite(os.path.join(out_dir, f"{base}_fixed.png"), thr_fixed)
        imwrite(os.path.join(out_dir, f"{base}_otsu.png"), thr_otsu)
        imwrite(os.path.join(out_dir, f"{base}_adapt.png"), thr_adapt)
        # Plot histograms with thresholds
        plt.figure(figsize=(6,4))
        plt.hist(img.ravel(), bins=256, range=(0,255), color='gray', alpha=0.6)
        plt.axvline(127, color='r', linestyle='--', label='fixed')
        plt.axvline(otsu_val, color='g', linestyle='--', label='otsu')
        plt.title(f"Histogram {base}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base}_hist.png"))
        plt.close()
        # Dice scores
        if mask is not None:
            for method, bin_img in [('fixed', thr_fixed), ('otsu', thr_otsu), ('adapt', thr_adapt)]:
                dice = dice_coef(bin_img, mask)
                rows.append({'image': fname, 'method': method, 'dice': dice})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "task8_metrics.csv"), index=False)
    print("Task 8 completed – metrics saved to task8_metrics.csv")

if __name__ == "__main__":
    main()
