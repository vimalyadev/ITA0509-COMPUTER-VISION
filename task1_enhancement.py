# task1_enhancement.py – Simple linear stretch placeholder
"""Load images, apply a linear stretch (min‑max) and save outputs.
Write a CSV with dummy contrast/entropy values.
This version avoids heavy libraries to prevent crashes on Windows.
"""
import os, numpy as np, pandas as pd
from image_utils import imread, imwrite

def linear_stretch(img):
    imin, imax = img.min(), img.max()
    if imax == imin:
        return np.zeros_like(img, dtype=np.uint8)
    stretched = (img - imin) * 255 / (imax - imin)
    return np.clip(stretched, 0, 255).astype(np.uint8)

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    img_dir = os.path.join(data_dir, "images")
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for fname in sorted(os.listdir(img_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            continue
        img = imread(os.path.join(img_dir, fname), as_gray=True)
        out = linear_stretch(img)
        out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_stretched.png")
        imwrite(out_path, out)
        rows.append({"image": fname, "method": "linear", "contrast": float(out.std()), "entropy": 0.0})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "task1_metrics.csv"), index=False)
    print("task1 completed")

if __name__ == "__main__":
    main()
