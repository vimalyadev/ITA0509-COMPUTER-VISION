"""
task2_arrays.py – Image Arrays, Coordinates, and Storage
==========================================================
Task 2 of Assignment 1: GreenHarvest Mango Grading
- a) Load image, print shape/dtype/channels/nbytes
- b) Coordinate probe: (x,y) → [row,col], BGR & RGB values
- c) Storage calculation vs ndarray.nbytes
- d) cv2.add vs NumPy uint8 addition on bright pixels
"""

import cv2, numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

with open("data/metadata.json") as f:
    META = json.load(f)

SAMPLE_PATH = os.path.join("data/images", META[0]["file"])

# ═══════════════════════════════════════════════════════════════════════════
# TASK 2-a  Image array properties
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 2-a  Image Array Properties")
print("=" * 60)

img = cv2.imread(SAMPLE_PATH)
assert img is not None, f"Cannot load {SAMPLE_PATH}"

h, w, c = img.shape
dtype    = img.dtype
nbytes   = img.nbytes

print(f"  File         : {SAMPLE_PATH}")
print(f"  Shape (H,W,C): {img.shape}")
print(f"  dtype        : {dtype}")
print(f"  Channels     : {c}")
print(f"  ndarray.nbytes: {nbytes:,} bytes  ({nbytes/1024:.2f} KB)")
print(f"  Pixel [0,0]  : BGR={img[0,0]}")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 2-b  Coordinate probe
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2-b  Coordinate Probe: (x,y) → [row, col]")
print("=" * 60)

# Probe a grid of displayed (x,y) positions
probe_points = [
    (w//4,   h//4),
    (w//2,   h//2),
    (3*w//4, h//4),
    (w//2,   3*h//4),
    (10,     10),
]

probe_rows = []
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"  {'Disp(x,y)':>14}  {'NumPy[row,col]':>16}  {'BGR':>18}  {'RGB':>18}")
for (x, y) in probe_points:
    row, col = y, x          # conversion: row=y, col=x
    bgr = img[row, col]
    rgb = img_rgb[row, col]
    print(f"  ({x:4d},{y:4d})  →  [{row:4d},{col:4d}]       "
          f"BGR={tuple(bgr)}   RGB={tuple(rgb)}")
    probe_rows.append({
        "disp_x": x, "disp_y": y, "row": row, "col": col,
        "B": int(bgr[0]), "G": int(bgr[1]), "R": int(bgr[2]),
        "R_rgb": int(rgb[0]), "G_rgb": int(rgb[1]), "B_rgb": int(rgb[2]),
    })

pd.DataFrame(probe_rows).to_csv("outputs/task2b_coordinate_probe.csv", index=False)
print("\nSaved → outputs/task2b_coordinate_probe.csv")

# Annotated image
annot = img.copy()
for (x, y) in probe_points:
    cv2.drawMarker(annot, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
    cv2.putText(annot, f"({x},{y})", (x+5, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)
cv2.imwrite("outputs/task2b_probed_image.png", annot)
print("Saved → outputs/task2b_probed_image.png")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 2-c  Storage calculation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2-c  Storage Calculation")
print("=" * 60)

bits_per_channel = 8    # uint8
calculated_bytes = h * w * c * (bits_per_channel // 8)
match = "✓ MATCH" if calculated_bytes == nbytes else "✗ MISMATCH"

print(f"  Image dimensions : {h} rows × {w} cols × {c} channels")
print(f"  Bit depth        : {bits_per_channel} bits/channel")
print(f"  Formula          : {h} × {w} × {c} × {bits_per_channel//8} = {calculated_bytes:,} bytes")
print(f"  ndarray.nbytes   : {nbytes:,} bytes")
print(f"  Result           : {match}")

for bd in [4, 6, 8, 12, 16]:
    b_per_px = max(1, bd // 8)
    total    = h * w * c * b_per_px
    print(f"  Bit depth {bd:2d}-bit → {total:,} bytes ({total/1024:.1f} KB)")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 2-d  cv2.add vs NumPy uint8 addition
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2-d  cv2.add vs NumPy uint8 addition (overflow comparison)")
print("=" * 60)

# Create a bright patch to force overflow
bright = img.copy()
# Make top-left quadrant very bright (200+)
bright[:h//2, :w//2] = np.clip(bright[:h//2, :w//2].astype(int) + 150, 0, 255).astype(np.uint8)

addend = np.full_like(bright, 80)  # add 80 to every pixel

cv2_result  = cv2.add(bright, addend)          # saturates at 255
np_result   = (bright.astype(np.uint16) + addend).astype(np.uint8)   # wraps for uint8
np_overflow = (bright + addend)                 # NumPy wraps modulo 256

# Find pixels that differ
diff_mask = cv2_result != np_overflow
diff_count = np.sum(diff_mask)
diff_pct   = 100.0 * diff_count / diff_mask.size
print(f"  Pixels differing (saturate vs wrap): {diff_count:,}  ({diff_pct:.2f}%)")

# Report first 10 differing pixels
rows_d, cols_d = np.where(diff_mask[:,:,0])
print(f"\n  First differing pixels (channel 0):")
print(f"  {'(row,col)':>12}  {'bright_px':>10}  {'cv2.add':>8}  {'np_wrap':>8}")
for i in range(min(10, len(rows_d))):
    r, c = rows_d[i], cols_d[i]
    print(f"  ({r:4d},{c:4d})       "
          f"{bright[r,c,0]:5d}     {cv2_result[r,c,0]:5d}    {np_overflow[r,c,0]:5d}")

# Save side-by-side
comparison = np.hstack([bright, cv2_result, np_overflow])
cv2.imwrite("outputs/task2d_addition_comparison.png", comparison)
print("\nSaved → outputs/task2d_addition_comparison.png")
print("  (Left=original bright, Centre=cv2.add, Right=NumPy wrap)")

# Matplotlib labeled figure
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name, img_s in zip(axes,
        ["Original (bright)", "cv2.add (saturate)", "NumPy uint8 (wrap)"],
        [cv2.cvtColor(bright, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(cv2_result, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(np_overflow, cv2.COLOR_BGR2RGB)]):
    ax.imshow(img_s)
    ax.set_title(name)
    ax.axis("off")
plt.suptitle("Task 2-d  Saturating vs Wrapping Addition", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task2d_comparison_plot.png", dpi=120)
plt.close()
print("Saved → outputs/task2d_comparison_plot.png")
print("\nTask 2 COMPLETE")
