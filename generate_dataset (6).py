# generate_dataset.py – Create a tiny synthetic snack‑packet dataset
"""Generates 30 synthetic snack‑packet images (solid coloured rectangles with a label).
The images are saved under ``data/images`` and are 600×800 (width×height) RGB.
No external data is required – this lets the rest of the scaffold run on any machine.
"""
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random


def create_packet(image_path, label, color):
    width, height = 600, 800
    img = Image.new('RGB', (width, height), color)
    draw = ImageDraw.Draw(img)
    # Write the label in the centre (simple sans‑serif font)
    try:
        font = ImageFont.truetype('arial.ttf', 80)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0,0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(((width - text_w) / 2, (height - text_h) / 2), label, fill='white', font=font)
    img.save(image_path)


def main():
    base_dir = Path(__file__).parent
    img_dir = base_dir / 'data' / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    colours = [
        (200, 30, 30), (30, 200, 30), (30, 30, 200),
        (200, 200, 30), (200, 30, 200), (30, 200, 200)
    ]
    for i in range(1, 31):
        label = f"SNACK_{i:02d}"
        colour = random.choice(colours)
        img_path = img_dir / f"{label}.png"
        create_packet(img_path, label, colour)
    print(f"Generated {i} synthetic packet images in {img_dir}")

if __name__ == '__main__':
    main()
