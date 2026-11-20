# task6_highpass_fft.py – Frequency‑domain high‑pass filters
"""Apply ideal, Butterworth and Gaussian high‑pass masks (two cut‑offs) to each page.
Save filtered images and a CSV of PSNR, SSIM and edge‑contrast metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import metrics
from image_utils import imread, imwrite, rgb_to_gray


def centred_fft(img: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(img)
    return np.fft.fftshift(f)


def magnitude_spectrum(fshift: np.ndarray) -> np.ndarray:
    mag = np.log1p(np.abs(fshift))
    mag = 255 * mag / mag.max()
    return mag.astype(np.uint8)


def ideal_highpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow) ** 2 + (j - ccol) ** 2 <= cutoff ** 2:
                mask[i, j] = 0
    return mask


def butterworth_highpass(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    mask = 1 / (1 + (cutoff / (distance + 1e-8)) ** (2 * order))
    return mask


def gaussian_highpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance_sq = (Y - crow) ** 2 + (X - ccol) ** 2
    mask = 1 - np.exp(-distance_sq / (2 * (cutoff ** 2)))
    return mask


def apply_filter(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fshift = centred_fft(img)
    fshift_f = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_f)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return np.clip(img_back, 0, 255).astype(np.uint8)


def edge_contrast(img: np.ndarray) -> float:
    gx = np.gradient(img.astype(np.float32), axis=1)
    gy = np.gradient(img.astype(np.float32), axis=0)
    grad = np.hypot(gx, gy)
    return float(grad.mean())


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
        for cutoff in [30, 60]:
            # Ideal
            mask = ideal_highpass(img.shape, cutoff)
            filtered = apply_filter(img, mask)
            imwrite(os.path.join(out_dir, f"{base}_ideal_hp_{cutoff}.png"), filtered)
            psnr = metrics.peak_signal_noise_ratio(img, filtered, data_range=255)
            ssim = metrics.structural_similarity(img, filtered, data_range=255)
            rows.append({'image': fname, 'type': 'ideal', 'cutoff': cutoff,
                         'psnr': psnr, 'ssim': ssim,
                         'edge_contrast': edge_contrast(filtered)})
            # Butterworth
            mask = butterworth_highpass(img.shape, cutoff, order=2)
            filtered = apply_filter(img, mask)
            imwrite(os.path.join(out_dir, f"{base}_butter_hp_{cutoff}.png"), filtered)
            psnr = metrics.peak_signal_noise_ratio(img, filtered, data_range=255)
            ssim = metrics.structural_similarity(img, filtered, data_range=255)
            rows.append({'image': fname, 'type': 'butterworth', 'cutoff': cutoff,
                         'psnr': psnr, 'ssim': ssim,
                         'edge_contrast': edge_contrast(filtered)})
            # Gaussian
            mask = gaussian_highpass(img.shape, cutoff)
            filtered = apply_filter(img, mask)
            imwrite(os.path.join(out_dir, f"{base}_gauss_hp_{cutoff}.png"), filtered)
            psnr = metrics.peak_signal_noise_ratio(img, filtered, data_range=255)
            ssim = metrics.structural_similarity(img, filtered, data_range=255)
            rows.append({'image': fname, 'type': 'gaussian', 'cutoff': cutoff,
                         'psnr': psnr, 'ssim': ssim,
                         'edge_contrast': edge_contrast(filtered)})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "task6_metrics.csv"), index=False)
    print("Task 6 completed – metrics saved to task6_metrics.csv")

if __name__ == "__main__":
    main()
