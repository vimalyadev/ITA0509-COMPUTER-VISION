"""
task5_quantization.py – Intensity Quantization
================================================
Task 5 of Assignment 1: GreenHarvest Mango Grading
- a) 8-bit, 6-bit, 4-bit quantization with NumPy; save images
- b) Local intensity contrast healthy peel vs defect before/after quantization
- c) Threshold each; compute defect precision and recall vs masks
- d) Plot bit depth vs storage, contrast, and recall; select minimum adequate
"""

import cv2, numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

with open("data/metadata.json") as f:
    META = json.load(f)

defect_meta = [m for m in META if m["defect"]]

def load_pair(m):
    img  = cv2.imread(os.path.join("data/images", m["file"]))
    mask = cv2.imread(os.path.join("data/images", m["mask"]), cv2.IMREAD_GRAYSCALE)
    return img, mask

BIT_DEPTHS = [8, 6, 4]

# ═══════════════════════════════════════════════════════════════════════════
# TASK 5-a  Quantization implementation and save
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 5-a  Intensity Quantization")
print("=" * 60)

QUANT_DIR = "outputs/task5a_quantized"
os.makedirs(QUANT_DIR, exist_ok=True)

def quantize(img: np.ndarray, bits: int) -> np.ndarray:
    """
    Reduce to `bits` intensity levels and scale back to 0-255.
    levels = 2^bits; step = 256 / levels
    """
    levels = 2 ** bits
    step   = 256 // levels
    q = (img.astype(np.uint16) // step) * step
    return np.clip(q, 0, 255).astype(np.uint8)

sample_img, sample_mask = load_pair(defect_meta[0])
q_images = {}
for bits in BIT_DEPTHS:
    q = quantize(sample_img, bits)
    q_images[bits] = q
    fname = f"task5a_quantized_{bits}bit.png"
    cv2.imwrite(f"outputs/{fname}", q)
    print(f"  {bits}-bit: levels={2**bits}, step={256//(2**bits)}, "
          f"saved → outputs/{fname}")

# Side-by-side figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, bits in zip(axes, BIT_DEPTHS):
    ax.imshow(cv2.cvtColor(q_images[bits], cv2.COLOR_BGR2RGB))
    ax.set_title(f"{bits}-bit ({2**bits} levels)")
    ax.axis("off")
plt.suptitle("Task 5-a  Intensity Quantization", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task5a_quantization_comparison.png", dpi=120)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TASK 5-b  Local contrast measurement
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 5-b  Local Contrast: Healthy Peel vs Defect")
print("=" * 60)

def measure_contrast(img_bgr, mask_gt):
    """Measure mean intensity difference between healthy peel and defect areas."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Peel without defect
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    peel_mask = cv2.inRange(hsv, (0,30,50),(180,255,255))
    healthy = peel_mask & (255 - mask_gt)   # peel AND not-defect
    defect  = peel_mask & mask_gt            # peel AND defect

    healthy_mean = gray[healthy > 0].mean() if np.any(healthy > 0) else 0
    defect_mean  = gray[defect  > 0].mean() if np.any(defect  > 0) else 0
    contrast     = abs(healthy_mean - defect_mean)
    return healthy_mean, defect_mean, contrast

contrast_rows = []
for bits in BIT_DEPTHS:
    q = quantize(sample_img, bits)
    h_m, d_m, cont = measure_contrast(q, sample_mask)
    h_orig, d_orig, cont_orig = measure_contrast(sample_img, sample_mask)
    contrast_rows.append({
        "bits": bits, "levels": 2**bits,
        "healthy_mean": round(h_m, 2), "defect_mean": round(d_m, 2),
        "contrast": round(cont, 2), "contrast_retention": round(cont/max(cont_orig,1)*100, 1),
    })
    print(f"  {bits}-bit: healthy={h_m:.1f}, defect={d_m:.1f}, "
          f"contrast={cont:.1f} ({cont/max(cont_orig,1)*100:.0f}% of original)")

h_m_orig, d_m_orig, cont_orig = measure_contrast(sample_img, sample_mask)
print(f"  8-bit original: healthy={h_m_orig:.1f}, defect={d_m_orig:.1f}, contrast={cont_orig:.1f}")

contrast_df = pd.DataFrame(contrast_rows)
contrast_df.to_csv("outputs/task5b_contrast.csv", index=False)
print("Saved → outputs/task5b_contrast.csv")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 5-c  Threshold + precision/recall vs masks
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 5-c  Defect Precision & Recall after Quantization")
print("=" * 60)

def detect_defects(img_bgr, thresh_val=90):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    peel = cv2.inRange(hsv, (0,30,50),(180,255,255))
    _, dark = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_and(dark, peel)

metrics_rows = []
for bits in BIT_DEPTHS:
    precisions, recalls = [], []
    for m in defect_meta:
        img, mask = load_pair(m)
        q    = quantize(img, bits)
        det  = detect_defects(q)
        tp   = np.sum((det > 0) & (mask > 0))
        fp   = np.sum((det > 0) & (mask == 0))
        fn   = np.sum((det == 0) & (mask > 0))
        prec = tp / (tp + fp + 1e-6)
        rec  = tp / (tp + fn + 1e-6)
        precisions.append(prec); recalls.append(rec)
    mp = np.mean(precisions); mr = np.mean(recalls)
    metrics_rows.append({"bits": bits, "precision": round(mp,4), "recall": round(mr,4)})
    print(f"  {bits}-bit: mean precision={mp:.4f}, mean recall={mr:.4f}")

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("outputs/task5c_defect_metrics.csv", index=False)
print("Saved → outputs/task5c_defect_metrics.csv")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 5-d  Plot bit depth vs storage, contrast, recall; select minimum
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 5-d  Bit Depth Trade-off Analysis")
print("=" * 60)

h, w = sample_img.shape[:2]
storage = [h * w * 3 * max(1, b // 8) for b in BIT_DEPTHS]  # bytes
contrast_ret = contrast_df["contrast_retention"].tolist()
recall_vals  = metrics_df["recall"].tolist()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].bar([str(b) for b in BIT_DEPTHS], storage, color="steelblue")
axes[0].set_xlabel("Bit depth"); axes[0].set_ylabel("Storage (bytes)")
axes[0].set_title("Storage vs Bit Depth")

axes[1].bar([str(b) for b in BIT_DEPTHS], contrast_ret, color="darkorange")
axes[1].set_xlabel("Bit depth"); axes[1].set_ylabel("Contrast retention (%)")
axes[1].set_title("Contrast Retention vs Bit Depth")

axes[2].bar([str(b) for b in BIT_DEPTHS], recall_vals, color="forestgreen")
axes[2].axhline(0.70, color="red", linestyle="--", label="Min recall=0.70")
axes[2].set_xlabel("Bit depth"); axes[2].set_ylabel("Defect Recall")
axes[2].set_title("Recall vs Bit Depth")
axes[2].legend()

plt.suptitle("Task 5-d  Bit Depth Trade-off", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task5d_bitdepth_tradeoff.png", dpi=120)
plt.close()
print("Saved → outputs/task5d_bitdepth_tradeoff.png")

# Selection: minimum bit depth with recall ≥ 0.70
RECALL_THRESHOLD = 0.70
selected_bits = None
for bits, rec in sorted(zip(BIT_DEPTHS, recall_vals)):
    if rec >= RECALL_THRESHOLD:
        selected_bits = bits
        break
print(f"\nSelected bit depth: {selected_bits}-bit  "
      f"(recall={dict(zip(BIT_DEPTHS,recall_vals))[selected_bits]:.4f} ≥ {RECALL_THRESHOLD})")
print("  → Minimum adequate bit depth that still achieves target recall.")
print("\nTask 5 COMPLETE")
