"""
task8_connectivity.py – Connectivity and Defect Measurements
=============================================================
Task 8 of Assignment 1: GreenHarvest Mango Grading
- a) Binary masks with isolated/touching/diagonal defects; 4- and 8-connectivity labels
- b) Component counts and areas vs ground truth
- c) Defect diameters: Euclidean, city-block, chessboard distance transforms
- d) Plot labels and distance maps; 4 mm rejection rule; FP/FN counts
"""

import cv2, numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt
from scipy import ndimage

os.makedirs("outputs", exist_ok=True)

MM_PER_PX   = 400.0 / 320      # from Task 4-b (selected 320px resolution)
REJECT_MM   = 4.0               # reject if defect diameter ≥ 4 mm
REJECT_PX   = REJECT_MM / MM_PER_PX

print(f"mm/px = {MM_PER_PX:.4f}  |  reject threshold = {REJECT_MM}mm = {REJECT_PX:.2f}px")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 8-a  Synthetic defect masks and connectivity labelling
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 8-a  Defect Masks – 4- and 8-connectivity")
print("=" * 60)

def make_mask():
    """
    200×200 binary mask with:
      - isolated defect (top-left)
      - two touching defects (centre)
      - diagonal-only connected defects (bottom-right)
    """
    m = np.zeros((200, 200), dtype=np.uint8)
    # Isolated (single blob)
    m[20:40, 20:50] = 255
    # Touching (two blobs share an edge pixel)
    m[90:110, 80:100] = 255
    m[90:110, 99:120] = 255          # shares column 99
    # Diagonal-only connected (two blobs touch only at diagonal)
    m[150:165, 140:155] = 255
    m[164:179, 154:169] = 255        # bottom-right corner of first = top-left of second
    return m

MASK = make_mask()
cv2.imwrite("outputs/task8a_synthetic_mask.png", MASK)
print("  Saved → outputs/task8a_synthetic_mask.png")

# Ground truth component info (manually annotated)
GT_COMPONENTS = {
    "4-connectivity": {"isolated":1, "touching":1, "diagonal":2},
    "8-connectivity": {"isolated":1, "touching":1, "diagonal":1},
}

for conn, conn_val, lbl_prefix in [
        ("4-connectivity", 4,  "task8a_label_4conn"),
        ("8-connectivity", 8,  "task8a_label_8conn")]:
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        MASK, connectivity=conn_val)
    n_components = num_labels - 1  # exclude background (label 0)
    print(f"\n  {conn}: {n_components} components (excl. background)")
    for ci in range(1, num_labels):
        area   = stats[ci, cv2.CC_STAT_AREA]
        cx, cy = centroids[ci]
        print(f"    Label {ci}: area={area}px², centroid=({cx:.1f},{cy:.1f})")

    # Colour map
    vis = np.zeros((*MASK.shape, 3), dtype=np.uint8)
    cmap = [(0,0,0),(200,50,50),(50,200,50),(50,50,200),(200,200,50),(200,50,200)]
    for li in range(1, min(num_labels, len(cmap))):
        vis[labels == li] = cmap[li]
    cv2.imwrite(f"outputs/{lbl_prefix}.png", vis)
    print(f"  Saved → outputs/{lbl_prefix}.png")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 8-b  Component count and area vs ground truth
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 8-b  Component Counts vs Ground Truth")
print("=" * 60)

results_b = []
for conn, conn_val in [("4-conn", 4), ("8-conn", 8)]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(MASK, connectivity=conn_val)
    n_found = num_labels - 1
    gt_conn = "4-connectivity" if conn == "4-conn" else "8-connectivity"
    gt_total = sum(GT_COMPONENTS[gt_conn].values())
    match = "✓" if n_found == gt_total else "✗"
    print(f"  {conn}: found={n_found}, GT={gt_total}  {match}")
    for li in range(1, num_labels):
        area_mm2 = stats[li, cv2.CC_STAT_AREA] * MM_PER_PX**2
        results_b.append({"connectivity":conn,"label":li,
                           "area_px":stats[li,cv2.CC_STAT_AREA],
                           "area_mm2":round(area_mm2,4)})

pd.DataFrame(results_b).to_csv("outputs/task8b_component_areas.csv", index=False)
print("Saved → outputs/task8b_component_areas.csv")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 8-c  Distance transforms
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 8-c  Distance Transforms – Euclidean, City-block, Chessboard")
print("=" * 60)

DIST_TYPES = [
    ("Euclidean",   cv2.DIST_L2,   cv2.DIST_MASK_PRECISE),
    ("City-block",  cv2.DIST_L1,   cv2.DIST_MASK_3),
    ("Chessboard",  cv2.DIST_C,    cv2.DIST_MASK_3),
]

dist_rows = []
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for col, (name, dtype, dmask) in enumerate(DIST_TYPES):
    dt = cv2.distanceTransform(MASK, dtype, dmask)
    # Maximum distance = approx half-width of largest component
    max_dist_px = dt.max()
    max_dist_mm = max_dist_px * MM_PER_PX
    diameter_px = 2 * max_dist_px
    diameter_mm = 2 * max_dist_mm
    dist_rows.append({"metric": name, "max_dist_px": round(max_dist_px,2),
                       "max_dist_mm": round(max_dist_mm,4),
                       "diameter_mm": round(diameter_mm,4)})
    print(f"  {name:12s}: max dist={max_dist_px:.2f}px = {max_dist_mm:.4f}mm → "
          f"diameter≈{diameter_mm:.4f}mm")

    axes[0][col].imshow(MASK, cmap="gray")
    axes[0][col].set_title(f"Mask ({name})"); axes[0][col].axis("off")
    im = axes[1][col].imshow(dt, cmap="jet")
    axes[1][col].set_title(f"{name} DT\nmax={max_dist_px:.1f}px ({max_dist_mm:.2f}mm)")
    axes[1][col].axis("off")
    plt.colorbar(im, ax=axes[1][col], fraction=0.046)

plt.suptitle("Task 8-c  Distance Transforms on Defect Mask", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task8c_distance_transforms.png", dpi=120)
plt.close()
print("Saved → outputs/task8c_distance_transforms.png")

pd.DataFrame(dist_rows).to_csv("outputs/task8c_distance_metrics.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# TASK 8-d  4 mm rejection rule; FP/FN counts
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 8-d  4 mm Rejection Rule (Euclidean DT)")
print("=" * 60)

# Euclidean DT
dt_euc = cv2.distanceTransform(MASK, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

# Label components (8-conn)
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(MASK, connectivity=8)

# Ground truth: defects with true diameter ≥ 4 mm (from stats & known mask)
# Annotated ground truth (for this synthetic mask):
#   Isolated (20×30 px) → ~24.5mm dia → REJECT
#   Touching (20×40 px) → merged ~32.5mm → REJECT
#   Diagonal (15×15 px each) → ~19.5mm → may or may not reject
GT_SHOULD_REJECT = {1: True, 2: True}   # GT: reject labels 1,2; label 3 borderline

reject_rows = []
FP, FN = 0, 0
for li in range(1, num_labels):
    dt_region = dt_euc.copy()
    dt_region[labels != li] = 0
    max_r_px   = dt_region.max()
    diameter_mm = 2 * max_r_px * MM_PER_PX
    should_reject = diameter_mm >= REJECT_MM
    gt_reject     = GT_SHOULD_REJECT.get(li, False)
    correct = "✓" if should_reject == gt_reject else "✗"
    if should_reject and not gt_reject:  FP += 1
    if not should_reject and gt_reject:  FN += 1
    reject_rows.append({"label": li, "diameter_mm": round(diameter_mm,3),
                         "rejected": should_reject, "gt_reject": gt_reject, "correct": correct})
    print(f"  Label {li}: dia={diameter_mm:.2f}mm → {'REJECT' if should_reject else 'PASS'}  "
          f"(GT={'REJECT' if gt_reject else 'PASS'})  {correct}")

print(f"\n  False Positives (rejected but OK): {FP}")
print(f"  False Negatives (passed but bad) : {FN}")

pd.DataFrame(reject_rows).to_csv("outputs/task8d_rejection_results.csv", index=False)

# Visualize label map + DT
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
vis = np.zeros((*MASK.shape, 3), dtype=np.uint8)
colors = [(0,0,0),(200,50,50),(50,200,50),(50,50,200)]
for li in range(1, num_labels):
    vis[labels==li] = colors[min(li,3)]
axes[0].imshow(vis); axes[0].set_title("8-conn Labels"); axes[0].axis("off")
axes[1].imshow(dt_euc, cmap="jet"); axes[1].set_title("Euclidean DT"); axes[1].axis("off")
plt.suptitle("Task 8-d  Label Map & Distance Transform", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task8d_labels_dt.png", dpi=120)
plt.close()
print("Saved → outputs/task8d_labels_dt.png, task8d_rejection_results.csv")
print("\nTask 8 COMPLETE")
