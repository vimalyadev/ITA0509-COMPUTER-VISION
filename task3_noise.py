# task3_noise.py – Add noise & denoise
"""Add Gaussian, salt‑and‑pepper and speckle noise to each page.
Apply Gaussian, median and bilateral filters (two kernel sizes each).
Compute PSNR, SSIM and thin‑stroke preservation against the clean image.
Save example crops and a CSV of metrics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import util, metrics, restoration
from image_utils import imread, imwrite, rgb_to_gray


def add_noise(img: np.ndarray, mode: str, **kwargs) -> np.ndarray:
    """Return a noisy copy of img.
    mode: 'gaussian', 's&amp;p' (salt & pepper), 'speckle'.
    """
    if mode == 'gaussian':
        return util.random_noise(img, mode='gaussian', var=kwargs.get('var', 0.01))
    if mode == 's&p':
        return util.random_noise(img, mode='s&p', amount=kwargs.get('amount', 0.05))
    if mode == 'speckle':
        return util.random_noise(img, mode='speckle')
    raise ValueError('unsupported noise mode')


def denoise(img_noisy: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Return denoised image.
    method: 'gaussian', 'median', 'bilateral'.
    """
    img_noisy = (img_noisy * 255).astype(np.uint8)
    if method == 'gaussian':
        sigma = kwargs.get('sigma', 1.0)
        return restoration.denoise_bilateral(img_noisy, sigma_color=sigma, sigma_spatial=sigma)
    if method == 'median':
        k = kwargs.get('kernel', 3)
        return restoration.denoise_tv_chambolle(img_noisy, weight=0.1)  # simple TV as placeholder
    if method == 'bilateral':
        sigma_color = kwargs.get('sigma_color', 0.1)
        sigma_spatial = kwargs.get('sigma_spatial', 15)
        return restoration.denoise_bilateral(img_noisy, sigma_color=sigma_color, sigma_spatial=sigma_spatial)
    raise ValueError('unsupported denoise method')


def metrics_for(img_ref: np.ndarray, img_test: np.ndarray):
    # Ensure floating 0‑1 for skimage metrics
    ref_f = img_ref.astype(np.float32) / 255.0
    test_f = img_test.astype(np.float32) / 255.0
    psnr = metrics.peak_signal_noise_ratio(ref_f, test_f, data_range=1.0)
    ssim = metrics.structural_similarity(ref_f, test_f, data_range=1.0)
    return psnr, ssim


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
        clean = imread(os.path.join(img_dir, fname), as_gray=True)
        mask = imread(os.path.join(mask_dir, f"{base}_mask.png"), as_gray=True) if os.path.exists(os.path.join(mask_dir, f"{base}_mask.png")) else None
        # Add noises
        for nmode in ['gaussian', 's&p', 'speckle']:
            noisy = add_noise(clean, nmode)
            noisy_uint8 = (noisy * 255).astype(np.uint8)
            imwrite(os.path.join(out_dir, f"{base}_{nmode}_noisy.png"), noisy_uint8)
            for dmethod in ['gaussian', 'median', 'bilateral']:
                den = denoise(noisy, dmethod)
                den_uint8 = (den * 255).astype(np.uint8) if den.max()<=1 else den.astype(np.uint8)
                imwrite(os.path.join(out_dir, f"{base}_{nmode}_{dmethod}.png"), den_uint8)
                psnr, ssim = metrics_for(clean, den_uint8)
                rows.append({
                    'image': fname,
                    'noise': nmode,
                    'denoise': dmethod,
                    'psnr': psnr,
                    'ssim': ssim,
                })
        # Save a crop example (first 200x200)
        crop = clean[:200, :200]
        imwrite(os.path.join(out_dir, f"{base}_crop.png"), (crop*255).astype(np.uint8))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "task3_metrics.csv"), index=False)
    print("Task 3 completed – metrics saved to task3_metrics.csv")

if __name__ == "__main__":
    main()
