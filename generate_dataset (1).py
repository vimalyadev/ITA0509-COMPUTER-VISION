"""
generate_dataset.py – Synthetic document pages for Assignment 2
==============================================================
Creates 30 page‑level images (800×800 px) with:
  * Random rotation (‑15° … 15°)
  * Soft shadow gradient (simulating uneven lighting)
  * Optional Gaussian blur (to emulate out‑of‑focus captures)
  * Random ink‑stain blobs
  * Black text drawn with OpenCV Hershey fonts (via Pillow for simplicity)

For every page a binary **text mask** (255 = text) is saved alongside the colour image.
All files are stored under `data/images/` and a `metadata.json` manifest is written.
"""

import os, json, random, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = "data/images"
os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (800, 800)
NUM_PAGES = 30
TEXT_LINES = 20  # per page
FONT_SIZE = 24
MAX_ROT = 15  # degrees
MAX_BLUR = 2.0  # Gaussian sigma (optional)
MAX_STAIN = 5   # number of random ink blobs per page

# Use a built‑in PIL font (fallback if Hershey not available)
try:
    FONT = ImageFont.truetype("arial.ttf", FONT_SIZE)
except Exception:
    FONT = ImageFont.load_default()

random.seed(2023)
np.random.seed(2023)

metadata = []
page_id = 0

for _ in range(NUM_PAGES):
    # ---- Base white canvas -------------------------------------------------
    img = Image.new("RGB", IMG_SIZE, color="white")
    mask = Image.new("L", IMG_SIZE, color=0)  # black background
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    # ---- Random rotation ---------------------------------------------------
    angle = random.uniform(-MAX_ROT, MAX_ROT)

    # ---- Add text lines ----------------------------------------------------
    line_h = IMG_SIZE[1] // (TEXT_LINES + 2)
    for i in range(TEXT_LINES):
        # Random short phrase (just letters for demo)
        txt = " ".join(random.choices(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), k=random.randint(10, 25)))
        y = (i + 1) * line_h
        x = random.randint(30, 100)
        draw_img.text((x, y), txt, font=FONT, fill="black")
        draw_mask.text((x, y), txt, font=FONT, fill=255)

    # ---- Apply rotation (about centre) ------------------------------------
    img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor="white")
    mask = mask.rotate(angle, resample=Image.NEAREST, expand=False, fillcolor=0)

    # ---- Soft shadow gradient (vertical) -----------------------------------
    shadow = Image.new("L", IMG_SIZE, color=0)
    for y in range(IMG_SIZE[1]):
        # Linear gradient from 0 (top) to 80 (bottom)
        val = int(80 * (y / IMG_SIZE[1]))
        ImageDraw.Draw(shadow).line([(0, y), (IMG_SIZE[0], y)], fill=val)
    shadow = shadow.filter(ImageFilter.GaussianBlur(200))
    # darken the image according to the shadow map
    img_np = np.array(img).astype(np.float32)
    shadow_np = np.array(shadow).astype(np.float32) / 255.0
    img_np = img_np * (1 - 0.4 * shadow_np)  # up to 40% darkening
    img = Image.fromarray(np.clip(img_np, 0, 255).astype(np.uint8))

    # ---- Random Gaussian blur (optional, small sigma) ---------------------
    if random.random() < 0.5:
        sigma = random.uniform(0.5, MAX_BLUR)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

    # ---- Random ink‑stain blobs -------------------------------------------
    draw_img = ImageDraw.Draw(img)
    for _ in range(random.randint(1, MAX_STAIN)):
        cx = random.randint(0, IMG_SIZE[0])
        cy = random.randint(0, IMG_SIZE[1])
        r = random.randint(10, 30)
        colour = (0, 0, 0)  # black ink
        draw_img.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)

    # ---- Save files --------------------------------------------------------
    img_name = f"page_{page_id:03d}.png"
    mask_name = f"mask_{page_id:03d}.png"
    img_path = os.path.join(OUT_DIR, img_name)
    mask_path = os.path.join(OUT_DIR, mask_name)
    img.save(img_path)
    mask.save(mask_path)

    metadata.append({
        "id": page_id,
        "file": img_name,
        "mask": mask_name,
        "rotation_deg": angle,
        "blur_sigma": sigma if 'sigma' in locals() else None,
        "size_px": IMG_SIZE,
    })
    page_id += 1

# ---------------------------------------------------------------------------
# Write manifest
# ---------------------------------------------------------------------------
with open("data/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Generated {NUM_PAGES} synthetic document pages → {OUT_DIR}/")
print("Metadata written to data/metadata.json")
