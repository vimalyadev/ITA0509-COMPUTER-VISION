"""
generate_dataset.py
-------------------
Generates 30+ synthetic mango images for the GreenHarvest assignment.
Saves to data/images/ with filenames: mango_{id:03d}_{label}.png
Labels: green, turning, ripe  |  Defects: clean, defect
"""

import cv2
import numpy as np
import os, json, random

OUT_DIR = "data/images"
os.makedirs(OUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)

LABELS   = ["green", "turning", "ripe"]
# BGR peel colours
BASE_BGR = {
    "green":   (40,  120,  40),
    "turning": (30,  160, 180),
    "ripe":    (20,   80, 230),
}

def draw_mango(label: str, has_defect: bool, seed: int) -> np.ndarray:
    """Draw a 300×300 synthetic mango with optional dark defect patches."""
    rng = np.random.default_rng(seed)
    img = np.zeros((300, 300, 3), dtype=np.uint8)

    # Background
    img[:] = (200, 200, 200)

    # Mango body — ellipse
    cx, cy = 150, 150
    a, b   = rng.integers(90, 120), rng.integers(70, 95)
    color  = tuple(int(c + rng.integers(-15, 15)) for c in BASE_BGR[label])
    cv2.ellipse(img, (cx, cy), (a, b), 0, 0, 360, color, -1)

    # Soft highlight
    hl_color = tuple(min(255, c + 60) for c in color)
    cv2.ellipse(img, (cx - a//4, cy - b//4), (a//3, b//3), -20, 0, 360, hl_color, -1)
    img = cv2.GaussianBlur(img, (25, 25), 8)

    # Stem
    cv2.line(img, (cx, cy - b), (cx + 10, cy - b - 25), (20, 60, 10), 4)

    if has_defect:
        for _ in range(rng.integers(1, 4)):
            dx = int(rng.integers(cx - a + 20, cx + a - 20))
            dy = int(rng.integers(cy - b + 20, cy + b - 20))
            dr = int(rng.integers(5, 18))
            dcolor = tuple(max(0, c - 60) for c in color)
            cv2.ellipse(img, (dx, dy), (dr, int(dr * 0.7)),
                        int(rng.integers(0, 180)), 0, 360, dcolor, -1)
        img = cv2.GaussianBlur(img, (5, 5), 1)

    return img


def draw_defect_mask(img_shape, label, has_defect, seed):
    """Binary mask of defect region (255=defect)."""
    rng = np.random.default_rng(seed + 1000)
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if not has_defect:
        return mask

    color = tuple(int(c) for c in BASE_BGR[label])
    # Reload the mango to find dark regions
    mango_img = draw_mango(label, has_defect, seed)
    # Defect pixels are darker than typical peel
    gray = cv2.cvtColor(mango_img, cv2.COLOR_BGR2GRAY)
    peel_mask_tmp = cv2.cvtColor(mango_img, cv2.COLOR_BGR2GRAY)
    # Simple threshold: dark spots on peel
    _, dark = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    # Remove background (very dark)
    kernel = np.ones((7, 7), np.uint8)
    dark   = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    mask   = dark
    return mask


metadata = []
img_id   = 0

for label in LABELS:
    for defect_flag in [False, True]:
        count = 5  # 5 clean + 5 defect × 3 labels = 30
        for i in range(count):
            seed  = img_id * 7 + i * 13
            img   = draw_mango(label, defect_flag, seed)
            fname = f"mango_{img_id:03d}_{label}_{'defect' if defect_flag else 'clean'}.png"
            fpath = os.path.join(OUT_DIR, fname)
            cv2.imwrite(fpath, img)

            mask  = draw_defect_mask(img.shape, label, defect_flag, seed)
            mname = f"mask_{img_id:03d}.png"
            cv2.imwrite(os.path.join(OUT_DIR, mname), mask)

            metadata.append({
                "id":      img_id,
                "file":    fname,
                "mask":    mname,
                "label":   label,
                "defect":  defect_flag,
                "width_px": img.shape[1],
                "height_px": img.shape[0],
            })
            img_id += 1

with open("data/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Generated {img_id} images → {OUT_DIR}/")
print(f"Metadata → data/metadata.json")
