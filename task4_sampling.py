"""
task4_sampling.py – Spatial Sampling and Defect Visibility
===========================================================
Task 4 of Assignment 1: GreenHarvest Mango Grading
- a) Resize to three resolutions using INTER_NEAREST and INTER_AREA
- b) Calculate mm/pixel and pixels across a 2 mm defect
- c) Defect recall vs annotated masks for each resize
- d) Plot defect crops; select resolution ≥ 4 px across smallest defect
"""

import cv2, numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

with open("data/metadata.json") as f:
    META = json.load(f)

FOV_MM    = 400.0        # 400 mm field of view
ORIG_W    = 300          # original image width in pixels
RESOLUTIONS = [(320, 240), (160, 120), (80, 60)]   # three target sizes
DEFECT_MM   = 2.0        # minimum defect width to detect

# ═══════════════════════════════════════════════════════════════════════════
# TASK 4-a  Resize with two interpolation methods
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 4-a  Resize samples – INTER_NEAREST vs INTER_AREA")
print("=" * 60)

RESIZE_DIR = "outputs/task4a_resized"
os.makedirs(RESIZE_DIR, exist_ok=True)

# Use defect images for visibility test
defect_meta = [m for m in META if m["defect"]][:3]

for m in defect_meta:
    img  = cv2.imread(os.path.join("data/images", m["file"]))
    mask = cv2.imread(os.path.join("data/images", m["mask"]), cv2.IMREAD_GRAYSCALE)
    for (tw, th) in RESOLUTIONS:
        for method, name in [(cv2.INTER_NEAREST, "NEAREST"), (cv2.INTER_AREA, "AREA")]:
            rim  = cv2.resize(img,  (tw, th), interpolation=method)
            rmas = cv2.resize(mask, (tw, th), interpolation=method)
            fname = f"id{m['id']:03d}_{tw}x{th}_{name}"
            cv2.imwrite(f"{RESIZE_DIR}/{fname}.png",  rim)
            cv2.imwrite(f"{RESIZE_DIR}/{fname}_mask.png", rmas)
print(f"Saved resized images → {RESIZE_DIR}/")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 4-b  mm/pixel and pixels per 2 mm defect
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 4-b  Scale Calculation: mm/pixel  &  px across 2 mm defect")
print("=" * 60)

# Original: FOV_MM / ORIG_W
scale_rows = []
print(f"  {'Resolution':>12}  {'mm/pixel':>10}  {'px per 2mm':>12}  {'Detectable':>12}")
for (tw, th) in [(ORIG_W, 300)] + RESOLUTIONS:
    mmpx = FOV_MM / tw
    px2mm = DEFECT_MM / mmpx
    detectable = "✓ YES" if px2mm >= 4 else "✗ NO"
    print(f"  {tw}×{th:>4}       {mmpx:9.4f}   {px2mm:11.2f}   {detectable}")
    scale_rows.append({"width_px": tw, "height_px": th, "mm_per_pixel": mmpx,
                        "px_per_2mm": px2mm, "detectable": detectable})

scale_df = pd.DataFrame(scale_rows)
scale_df.to_csv("outputs/task4b_scale_calculations.csv", index=False)
print("Saved → outputs/task4b_scale_calculations.csv")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 4-c  Defect recall and apparent area
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 4-c  Defect Recall vs Annotated Masks")
print("=" * 60)

def measure_recall(orig_img, orig_mask, tw, th, method):
    """Resize and threshold; compute recall vs resized GT mask."""
    rim   = cv2.resize(orig_img,  (tw, th), interpolation=method)
    rmask = cv2.resize(orig_mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    # Detect dark patches as defects
    gray  = cv2.cvtColor(rim, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold to detect dark spots
    _, thresh = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)
    # Mask to peel area only
    hsv   = cv2.cvtColor(rim, cv2.COLOR_BGR2HSV)
    peel  = cv2.inRange(hsv, (0,30,50),(180,255,255))
    detected = cv2.bitwise_and(thresh, peel)

    gt_pos   = np.sum(rmask > 0)
    detected_pos = np.sum(detected > 0)
    tp = np.sum((detected > 0) & (rmask > 0))
    recall = tp / gt_pos if gt_pos > 0 else 0.0
    area_ratio = detected_pos / (gt_pos + 1e-6)
    return recall, area_ratio

recall_rows = []
for m in defect_meta:
    orig_img  = cv2.imread(os.path.join("data/images", m["file"]))
    orig_mask = cv2.imread(os.path.join("data/images", m["mask"]), cv2.IMREAD_GRAYSCALE)
    for (tw, th) in RESOLUTIONS:
        for method, mname in [(cv2.INTER_NEAREST,"NEAREST"),(cv2.INTER_AREA,"AREA")]:
            recall, area_r = measure_recall(orig_img, orig_mask, tw, th, method)
            recall_rows.append({
                "id": m["id"], "resolution": f"{tw}x{th}",
                "method": mname, "recall": round(recall,4), "area_ratio": round(area_r,4),
            })

recall_df = pd.DataFrame(recall_rows)
recall_df.to_csv("outputs/task4c_defect_recall.csv", index=False)
print(recall_df.to_string(index=False))
print("Saved → outputs/task4c_defect_recall.csv")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 4-d  Defect crop plots; resolution selection
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 4-d  Defect Crop Plots & Resolution Selection")
print("=" * 60)

m = defect_meta[0]
orig_img = cv2.imread(os.path.join("data/images", m["file"]))

rows_fig = len(RESOLUTIONS)
fig, axes = plt.subplots(rows_fig, 3, figsize=(12, rows_fig*3))
for ri, (tw, th) in enumerate(RESOLUTIONS):
    mmpx  = FOV_MM / tw
    px2mm = DEFECT_MM / mmpx
    rim_n = cv2.resize(orig_img, (tw,th), interpolation=cv2.INTER_NEAREST)
    rim_a = cv2.resize(orig_img, (tw,th), interpolation=cv2.INTER_AREA)
    for ci, (rim, label) in enumerate([(orig_img,"Original (300×300)"),
                                        (rim_n, "INTER_NEAREST"),
                                        (rim_a, "INTER_AREA")]):
        ax = axes[ri][ci]
        ax.imshow(cv2.cvtColor(rim, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\n{tw}×{th}, {mmpx:.2f} mm/px, 2mm≈{px2mm:.1f}px", fontsize=8)
        ax.axis("off")

plt.suptitle("Task 4-d  Defect Crops at Different Resolutions", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task4d_defect_crops.png", dpi=120)
plt.close()
print("Saved → outputs/task4d_defect_crops.png")

# Selection
best_res = None
for (tw, th) in [(ORIG_W, 300)] + RESOLUTIONS:
    mmpx  = FOV_MM / tw
    px2mm = DEFECT_MM / mmpx
    if px2mm >= 4:
        best_res = (tw, th, mmpx, px2mm)
        break  # coarsest that still passes (last iteration gives finest; take finest passing)

# Actually select the finest passing
passing = [(tw,th, FOV_MM/tw, (DEFECT_MM/(FOV_MM/tw)))
           for (tw,th) in [(ORIG_W,300)]+RESOLUTIONS if (DEFECT_MM/(FOV_MM/tw))>=4]
if passing:
    tw,th,mmpx,px2mm = passing[-1]
    print(f"Selected resolution: {tw}×{th}  →  {mmpx:.4f} mm/px  "
          f"→  {px2mm:.1f} px per 2 mm defect  (≥4 px requirement ✓)")
else:
    print("No resolution passes the 4-px threshold — increase sensor resolution.")

print("\nTask 4 COMPLETE")
