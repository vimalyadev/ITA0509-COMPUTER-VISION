"""
task1_pipeline.py – Vision Systems and Processing Levels
=========================================================
Task 1 of Assignment 1: GreenHarvest Mango Grading
- a) Low / mid / high level pipeline with printed representations
- b) Manual-label, image-processing-assisted, rule-based grading → CSV
- c) Agreement, confusion matrices, Cohen's kappa
- d) Latency and fruits-per-second measurement
"""

import cv2, numpy as np, pandas as pd, time, os, json
from sklearn.metrics import confusion_matrix, cohen_kappa_score, classification_report
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

# ── Load dataset ────────────────────────────────────────────────────────────
with open("data/metadata.json") as f:
    META = json.load(f)

def load_images():
    imgs, labels, ids = [], [], []
    for m in META:
        p = os.path.join("data/images", m["file"])
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img)
            labels.append(m["label"])
            ids.append(m["id"])
    return imgs, labels, ids

IMAGES, GT_LABELS, IDS = load_images()
N = len(IMAGES)
print(f"Loaded {N} images")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 1-a  Low / Mid / High level pipeline
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 1-a  Three-Level Vision Pipeline")
print("="*60)

def low_level(img: np.ndarray) -> dict:
    """Raw pixel operations: resize, blur, edge detection."""
    resized = cv2.resize(img, (128, 128))
    gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    edges   = cv2.Canny(blurred, 30, 100)
    print(f"  [LOW]  Input: ndarray {img.shape} uint8 | "
          f"Output: edges ndarray {edges.shape} uint8, "
          f"nonzero={np.count_nonzero(edges)}")
    return {"resized": resized, "gray": gray, "edges": edges, "blurred": blurred}

def mid_level(low: dict) -> dict:
    """Segmentation and region analysis."""
    # HSV-based segmentation to isolate the mango body
    hsv    = cv2.cvtColor(low["resized"], cv2.COLOR_BGR2HSV)
    # Peel mask: exclude grey background
    mask   = cv2.inRange(hsv, (0, 30, 50), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Region properties
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area    = max((cv2.contourArea(c) for c in cnts), default=0)
    peel_px = cv2.mean(low["resized"], mask=mask)[:3]  # mean BGR on peel
    print(f"  [MID]  Input: resized {low['resized'].shape} | "
          f"Output: mask {mask.shape}, peel area={area:.0f}px², mean BGR={np.round(peel_px,1)}")
    return {"mask": mask, "peel_area": area, "mean_bgr": peel_px, "contours": cnts}

def high_level(mid: dict) -> str:
    """Semantic decision: ripeness classification."""
    b, g, r = mid["mean_bgr"]
    # Rule: ripe → high R; green → high G; turning → intermediate
    if r > 160:
        decision = "ripe"
    elif g > r and g > b:
        decision = "green"
    else:
        decision = "turning"
    print(f"  [HIGH] Input: mean BGR=({b:.1f},{g:.1f},{r:.1f}) | "
          f"Output: label='{decision}'")
    return decision

sample = IMAGES[0]
print(f"\nRunning pipeline on sample image (id=0):")
ll = low_level(sample)
ml = mid_level(ll)
hl = high_level(ml)
print(f"\n  → Final decision: '{hl}'  (ground truth: '{GT_LABELS[0]}')")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 1-b  Three grading approaches
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 1-b  Grading Approaches")
print("="*60)

# --- Approach 1: "Manual" label (simulated from ground truth with noise) ---
rng = np.random.default_rng(99)
label_map = {"green": 0, "turning": 1, "ripe": 2}
rev_map   = {v: k for k, v in label_map.items()}

def approach_manual(labels):
    """Simulate manual labelling with 10% random error."""
    out = []
    for l in labels:
        if rng.random() < 0.10:
            out.append(rev_map[rng.integers(0, 3)])
        else:
            out.append(l)
    return out

# --- Approach 2: Image-processing-assisted ---
def approach_ip(images):
    """HSV mean-based heuristic (mid/high pipeline)."""
    preds = []
    for img in images:
        ll = low_level.__wrapped__(img) if hasattr(low_level, "__wrapped__") else {"resized": cv2.resize(img,(128,128))}
        resized = ll.get("resized", cv2.resize(img,(128,128)))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0,30,50),(180,255,255))
        mean = cv2.mean(resized, mask=mask)[:3]
        b,g,r = mean
        if r > 160:   preds.append("ripe")
        elif g > r:   preds.append("green")
        else:          preds.append("turning")
    return preds

# --- Approach 3: Rule-based (HSV hue threshold) ---
def approach_rule(images):
    """Hue-based rule set."""
    preds = []
    for img in images:
        resized = cv2.resize(img,(128,128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,(0,30,50),(180,255,255))
        mean_hsv = cv2.mean(hsv, mask=mask)
        h = mean_hsv[0]
        if h < 30 or h > 150:   preds.append("ripe")    # orange/red hues
        elif h < 70:              preds.append("turning")
        else:                     preds.append("green")
    return preds

manual_preds = approach_manual(GT_LABELS)
ip_preds     = approach_ip(IMAGES)
rule_preds   = approach_rule(IMAGES)

# Save CSV
df = pd.DataFrame({
    "id":          IDS,
    "ground_truth": GT_LABELS,
    "manual":      manual_preds,
    "ip_assisted": ip_preds,
    "rule_based":  rule_preds,
})
df.to_csv("outputs/task1_grading_decisions.csv", index=False)
print("Saved → outputs/task1_grading_decisions.csv")
print(df.head(10).to_string(index=False))

# ═══════════════════════════════════════════════════════════════════════════
# TASK 1-c  Agreement, Confusion Matrices, Cohen's Kappa
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 1-c  Agreement & Cohen's Kappa")
print("="*60)

classes = ["green","turning","ripe"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, preds) in zip(axes, [
        ("Manual",      manual_preds),
        ("IP-Assisted", ip_preds),
        ("Rule-Based",  rule_preds)]):
    cm    = confusion_matrix(GT_LABELS, preds, labels=classes)
    kappa = cohen_kappa_score(GT_LABELS, preds)
    acc   = np.sum(np.diag(cm)) / np.sum(cm)
    print(f"\n{name}:")
    print(f"  Accuracy: {acc:.3f}  |  Cohen's κ: {kappa:.3f}")
    print(f"  Confusion matrix:\n{cm}")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{name}\nAccuracy={acc:.2f}, κ={kappa:.2f}")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")

plt.suptitle("Task 1-c  Confusion Matrices – Three Grading Approaches", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/task1c_confusion_matrices.png", dpi=120)
plt.close()
print("\nSaved → outputs/task1c_confusion_matrices.png")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 1-d  Latency Measurement
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 1-d  Latency & Throughput")
print("="*60)

TARGET_FPS = 8.0  # 8 fruits/s requirement

def bench(fn, *args, reps=5):
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        times.append(time.perf_counter() - t0)
    return np.mean(times)

# Time per image for each approach
t_manual = bench(approach_manual, GT_LABELS)       # whole batch
t_ip     = bench(approach_ip,     IMAGES)
t_rule   = bench(approach_rule,   IMAGES)

latency_rows = []
for name, t_total in [("Manual", t_manual), ("IP-Assisted", t_ip), ("Rule-Based", t_rule)]:
    per_fruit = t_total / N
    fps       = 1.0 / per_fruit
    meets     = "✓ YES" if fps >= TARGET_FPS else "✗ NO"
    latency_rows.append({
        "Approach":          name,
        "Total_s":           round(t_total, 4),
        "Latency_per_fruit": round(per_fruit * 1000, 3),   # ms
        "Fruits_per_s":      round(fps, 2),
        "Meets_8fps":        meets,
    })
    print(f"  {name:14s}: {per_fruit*1000:.2f} ms/fruit → {fps:.1f} fps  {meets}")

lat_df = pd.DataFrame(latency_rows)
lat_df.to_csv("outputs/task1d_latency.csv", index=False)
print("\nSaved → outputs/task1d_latency.csv")

# Bar chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(lat_df["Approach"], lat_df["Fruits_per_s"], color=["steelblue","darkorange","forestgreen"])
ax.axhline(TARGET_FPS, color="red", linestyle="--", label=f"Target {TARGET_FPS} fps")
ax.set_ylabel("Fruits per second")
ax.set_title("Task 1-d  Throughput vs Target")
ax.legend()
plt.tight_layout()
plt.savefig("outputs/task1d_throughput.png", dpi=120)
plt.close()
print("Saved → outputs/task1d_throughput.png")
print("\nTask 1 COMPLETE")
