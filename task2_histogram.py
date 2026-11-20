# task2_histogram.py – Simple global histogram equalization
"""Apply a basic global histogram equalization to each page image.
Save the equalised image and a CSV with a dummy contrast metric.
"""

import os, numpy as np, pandas as pd
from image_utils import imread, imwrite

def hist_equalize(img: np.ndarray) -> np.ndarray:
    # Compute histogram and cumulative distribution
    hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 255))
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min() + 1e-8)
    img_eq = np.interp(img.ravel(), bins[:-1], cdf_normalized).reshape(img.shape).astype(np.uint8)
    return img_eq

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
        eq = hist_equalize(img)
        imwrite(os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_eq.png"), eq)
        rows.append({"image": fname, "method": "global_eq", "contrast": float(eq.std())})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "task2_metrics.csv"), index=False)
    print("task2 completed")

if __name__ == "__main__":
    main()
