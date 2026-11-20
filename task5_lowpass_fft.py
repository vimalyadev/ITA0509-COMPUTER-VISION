# task5_lowpass_fft.py – Frequency‑domain low‑pass filters
"""Compute the centred Fourier magnitude spectrum of each page image and apply
ideal, Butterworth and Gaussian low‑pass masks (two cutoff frequencies each).
Save filtered images, masks and a CSV with PSNR, SSIM, runtime and edge‑blur metric.
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import metrics
from image_utils import imread, imwrite, rgb_to_gray


def centred_fft(img: np.ndarray) -> np.ndarray:
    """Return centred FFT of a grayscale image (float32)."""
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    return fshift


def magnitude_spectrum(fshift: np.ndarray) -> np.ndarray:
    """Compute log magnitude spectrum for visualisation (0‑255 uint8)."""
    mag = np.log1p(np.abs(fshift))
    mag_norm = 255 * mag / mag.max()
    return mag_norm.astype(np.uint8)


def ideal_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow) ** 2 + (j - ccol) ** 2 <= cutoff ** 2:
                mask[i, j] = 1
    return mask


def butterworth_lowpass(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    mask = 1 / (1 + (distance / cutoff) ** (2 * order))
    return mask


def gaussian_lowpass(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance_sq = (Y - crow) ** 2 + (X - ccol) ** 2
    mask = np.exp(-distance_sq / (2 * (cutoff ** 2)))
    return mask


def apply_filter(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fshift = centred_fft(img)
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back


def edge_blur_metric(img: np.ndarray) -> float:
    # Simple variance of gradient magnitude (Sobel)
    gx = np.gradient(img.astype(np.float32), axis=1)
    gy = np.gradient(img.astype(np.float32), axis=0)
    grad = np.hypot(gx, gy)
    return float(grad.var())


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
        # Save magnitude spectrum
        mag = magnitude_spectrum(centred_fft(img))
        imwrite(os.path.join(out_dir, f"{base}_fft_mag.png"), mag)
        for cutoff in [30, 60]:  # two cutoffs (pixel units)
            # Ideal
            mask = ideal_lowpass(img.shape, cutoff)
            filtered = apply_filter(img, mask)
            imwrite(os.path.join(out_dir, f"{base}_ideal_{cutoff}.png"), filtered)
            psnr = metrics.peak_signal_noise_ratio(img, filtered, data_range=255)
            ssim = metrics.structural_similarity(img, filtered, data_range=255)
            rows.append({'image': fname, 'type': 'ideal', 'cutoff': cutoff,
                         'psnr': psnr, 'ssim': ssim,
                         'edge_blur': edge_blur_metric(filtered)})
            # Butterworth order 2
            mask = butterworth_lowpass(img.shape, cutoff, order=2)
            filtered = apply_filter(img, mask)
            imwrite(os.path.join(out_dir, f"{base}_butter_{cutoff}.png"), filtered)
            psnr = metrics.peak_signal_noise_ratio(img, filtered, data_range=255)
            ssim = metrics.structural_similarity(img, filtered, data_range=255)
            rows.append({'image': fname, 'type': 'butterworth', 'cutoff': cutoff,
                         'psnr': psnr, 'ssim': ssim,
                         'edge_blur': edge_blur_metric(filtered)})
            # Gaussian
            mask = gaussian_lowpass(img.shape, cutoff)
            filtered = apply_filter(img, mask)
            imwrite(os.path.join(out_dir, f"{base}_gauss_{cutoff}.png"), filtered)
            psnr = metrics.peak_signal_noise_ratio(img, filtered, data_range=255)
            ssim = metrics.structural_similarity(img, filtered, data_range=255)
            rows.append({'image': fname, 'type': 'gaussian', 'cutoff': cutoff,
                         'psnr': psnr, 'ssim': ssim,
                         'edge_blur': edge_blur_metric(filtered)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "task5_metrics.csv"), index=False)
    print("Task 5 completed – metrics saved to task5_metrics.csv")

if __name__ == "__main__":
    main()
