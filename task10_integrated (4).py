# task10_integrated.py – Integrated summary of Assignment 2
"""Load per‑task metric CSV files, compute average scores per method, and
write a concise summary CSV. This file is optional – run_all.py already runs
all tasks, but the final report generator expects this summary.
"""

import os
import pandas as pd


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    summary_rows = []
    # Task 1 – contrast & entropy (higher better)
    t1 = load_csv(os.path.join(out_dir, "task1_metrics.csv"))
    if t1 is not None:
        for method, grp in t1.groupby('method'):
            summary_rows.append({
                'task': '1_enhancement',
                'method': method,
                'mean_contrast': grp['contrast'].mean(),
                'mean_entropy': grp['entropy'].mean(),
                'mean_recall': grp.get('text_recall', pd.Series([None])).mean()
            })
    # Task 2 – global/CLAHE (contrast, entropy, recall)
    t2 = load_csv(os.path.join(out_dir, "task2_metrics.csv"))
    if t2 is not None:
        for method, grp in t2.groupby('method'):
            summary_rows.append({
                'task': '2_histogram',
                'method': method,
                'mean_contrast': grp['contrast'].mean(),
                'mean_entropy': grp['entropy'].mean(),
                'mean_recall': grp.get('text_recall', pd.Series([None])).mean()
            })
    # Task 3 – noise/denoise (psnr, ssim)
    t3 = load_csv(os.path.join(out_dir, "task3_metrics.csv"))
    if t3 is not None:
        for (noise, denoise), grp in t3.groupby(['noise', 'denoise']):
            summary_rows.append({
                'task': '3_noise',
                'method': f"{noise}_{denoise}",
                'mean_psnr': grp['psnr'].mean(),
                'mean_ssim': grp['ssim'].mean()
            })
    # Task 4 – sharpening (sharpness)
    t4 = load_csv(os.path.join(out_dir, "task4_metrics.csv"))
    if t4 is not None:
        for method, grp in t4.groupby('method'):
            summary_rows.append({
                'task': '4_sharpen',
                'method': method,
                'mean_sharpness': grp['sharpness'].mean()
            })
    # Task 5 – low‑pass (psnr, ssim, edge_blur)
    t5 = load_csv(os.path.join(out_dir, "task5_metrics.csv"))
    if t5 is not None:
        for (ftype, cutoff), grp in t5.groupby(['type', 'cutoff']):
            summary_rows.append({
                'task': '5_lowpass',
                'method': f"{ftype}_{cutoff}",
                'mean_psnr': grp['psnr'].mean(),
                'mean_ssim': grp['ssim'].mean(),
                'mean_edge_blur': grp['edge_blur'].mean()
            })
    # Task 6 – high‑pass (psnr, ssim, edge_contrast)
    t6 = load_csv(os.path.join(out_dir, "task6_metrics.csv"))
    if t6 is not None:
        for (ftype, cutoff), grp in t6.groupby(['type', 'cutoff']):
            summary_rows.append({
                'task': '6_highpass',
                'method': f"{ftype}_{cutoff}",
                'mean_psnr': grp['psnr'].mean(),
                'mean_ssim': grp['ssim'].mean(),
                'mean_edge_contrast': grp['edge_contrast'].mean()
            })
    # Task 7 – edge detection (precision, recall, f1)
    t7 = load_csv(os.path.join(out_dir, "task7_metrics.csv"))
    if t7 is not None:
        for method, grp in t7.groupby('method'):
            summary_rows.append({
                'task': '7_edge',
                'method': method,
                'mean_precision': grp['precision'].mean(),
                'mean_recall': grp['recall'].mean(),
                'mean_f1': grp['f1'].mean()
            })
    # Task 8 – thresholding (dice)
    t8 = load_csv(os.path.join(out_dir, "task8_metrics.csv"))
    if t8 is not None:
        for method, grp in t8.groupby('method'):
            summary_rows.append({
                'task': '8_threshold',
                'method': method,
                'mean_dice': grp['dice'].mean()
            })
    # Task 9 – segmentation (pixel_acc, iou, dice)
    t9 = load_csv(os.path.join(out_dir, "task9_metrics.csv"))
    if t9 is not None:
        for _, row in t9.iterrows():
            summary_rows.append({
                'task': '9_segmentation',
                'method': 'binary',
                'pixel_acc': row['pixel_acc'],
                'iou': row['iou'],
                'dice': row['dice']
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Integrated summary written to {summary_path}")

if __name__ == "__main__":
    main()
