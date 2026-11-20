"""
task7_illumination.py – Illumination and Exposure
===================================================
Task 7 of Assignment 1: GreenHarvest Mango Grading
- a) Motion blur calculation for 0.1, 1, 4 ms at 0.5 m/s
- b) Exposure/brightness sweep; save comparable fruit images
- c) Illumination uniformity for diffuse, side, and ring-light simulations
- d) Select exposure and lighting; save parameters to JSON
"""

import cv2, numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt

os.makedirs("outputs", exist_ok=True)

with open("data/metadata.json") as f:
    META = json.load(f)

sample_path = os.path.join("data/images", META[2]["file"])
SAMPLE_IMG  = cv2.imread(sample_path)

BELT_SPEED_MS = 0.5   # m/s
FOV_MM        = 400.0
SENSOR_PX     = 320

# ═══════════════════════════════════════════════════════════════════════════
# TASK 7-a  Motion blur calculation
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 7-a  Motion Blur vs Exposure Time")
print("=" * 60)

EXPOSURES_MS = [0.1, 1.0, 4.0]   # milliseconds
mm_per_px    = FOV_MM / SENSOR_PX

blur_rows = []
print(f"  Belt speed: {BELT_SPEED_MS} m/s  |  mm/px: {mm_per_px:.4f}")
print(f"  {'Exposure (ms)':>14}  {'Blur (mm)':>10}  {'Blur (px)':>10}  {'Acceptable?':>12}")
for exp_ms in EXPOSURES_MS:
    exp_s    = exp_ms * 1e-3
    blur_mm  = BELT_SPEED_MS * 1000 * exp_s    # mm
    blur_px  = blur_mm / mm_per_px
    ok       = "✓ <0.5px" if blur_px < 0.5 else ("⚠ <1px" if blur_px < 1 else "✗ >1px")
    print(f"  {exp_ms:>14}  {blur_mm:>10.4f}  {blur_px:>10.4f}  {ok:>12}")
    blur_rows.append({"exposure_ms": exp_ms, "blur_mm": round(blur_mm,4),
                       "blur_px": round(blur_px,4), "acceptable": ok})

blur_df = pd.DataFrame(blur_rows)
blur_df.to_csv("outputs/task7a_motion_blur.csv", index=False)
print("Saved → outputs/task7a_motion_blur.csv")

# Visualize blur effect
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, row in zip(axes, blur_rows):
    blur_px_int = max(1, int(row["blur_px"] * 2) | 1)   # kernel must be odd
    kernel = np.zeros((blur_px_int, blur_px_int))
    kernel[blur_px_int//2, :] = 1.0 / blur_px_int         # horizontal motion
    blurred = cv2.filter2D(SAMPLE_IMG, -1, kernel)
    ax.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{row['exposure_ms']} ms\nblur={row['blur_mm']:.3f}mm / {row['blur_px']:.3f}px")
    ax.axis("off")
plt.suptitle("Task 7-a  Motion Blur Simulation", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task7a_motion_blur_simulation.png", dpi=120)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TASK 7-b  Exposure / brightness sweep
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 7-b  Exposure / Brightness Sweep")
print("=" * 60)

SWEEP_DIR = "outputs/task7b_exposure_sweep"
os.makedirs(SWEEP_DIR, exist_ok=True)

GAMMA_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0]   # simulates different exposures
sweep_rows   = []

for g in GAMMA_VALUES:
    # Apply gamma to simulate exposure change
    table = np.array([(i/255.0)**(1.0/g)*255 for i in range(256)], dtype=np.uint8)
    exposed = cv2.LUT(SAMPLE_IMG, table)
    fname   = f"exposure_gamma{str(g).replace('.','p')}.png"
    cv2.imwrite(f"{SWEEP_DIR}/{fname}", exposed)
    mean_int = exposed.mean()
    sweep_rows.append({"gamma": g, "mean_intensity": round(mean_int,2), "file": fname})
    print(f"  gamma={g}: mean intensity={mean_int:.1f}  → saved {fname}")

pd.DataFrame(sweep_rows).to_csv("outputs/task7b_sweep.csv", index=False)
print("Saved → outputs/task7b_sweep.csv")

# Comparison figure
fig, axes = plt.subplots(1, len(GAMMA_VALUES), figsize=(15, 3))
for ax, row in zip(axes, sweep_rows):
    img_s = cv2.imread(f"{SWEEP_DIR}/{row['file']}")
    ax.imshow(cv2.cvtColor(img_s, cv2.COLOR_BGR2RGB))
    ax.set_title(f"γ={row['gamma']}\nmean={row['mean_intensity']:.0f}", fontsize=8)
    ax.axis("off")
plt.suptitle("Task 7-b  Exposure Sweep (gamma simulation)", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task7b_exposure_sweep.png", dpi=120)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════
# TASK 7-c  Illumination variation measurement
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 7-c  Illumination Uniformity")
print("=" * 60)

h, w = SAMPLE_IMG.shape[:2]

def simulate_diffuse(img) -> np.ndarray:
    """Uniform Gaussian blur = soft diffuse illumination."""
    return cv2.GaussianBlur(img, (1,1), 0)

def simulate_side_light(img) -> np.ndarray:
    """Left-side gradient – brighter on left."""
    grad = np.linspace(1.4, 0.6, w, dtype=np.float32)
    grad = np.tile(grad, (h, 1))
    lit  = np.clip(img.astype(np.float32) * grad[:,:,None], 0, 255).astype(np.uint8)
    return lit

def simulate_ring_light(img) -> np.ndarray:
    """Vignette effect – bright centre, dark edges."""
    cy, cx = h//2, w//2
    Y, X   = np.ogrid[:h, :w]
    dist   = np.sqrt((X - cx)**2 + (Y - cy)**2)
    ring   = np.clip(1.0 - dist / (max(h,w) * 0.7), 0.5, 1.5).astype(np.float32)
    lit    = np.clip(img.astype(np.float32) * ring[:,:,None], 0, 255).astype(np.uint8)
    return lit

light_sims = {
    "Diffuse":    simulate_diffuse(SAMPLE_IMG),
    "Side-light": simulate_side_light(SAMPLE_IMG),
    "Ring-light": simulate_ring_light(SAMPLE_IMG),
}

illum_rows = []
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for col, (name, sim_img) in enumerate(light_sims.items()):
    gray = cv2.cvtColor(sim_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean_int   = gray.mean()
    std_int    = gray.std()
    cv_percent = std_int / mean_int * 100   # coefficient of variation

    cv2.imwrite(f"outputs/task7c_{name.lower().replace('-','_')}.png", sim_img)
    illum_rows.append({"Light": name, "Mean_intensity": round(mean_int,2),
                        "Std_intensity": round(std_int,2), "CV_pct": round(cv_percent,2)})
    print(f"  {name:12s}: mean={mean_int:.1f}, std={std_int:.1f}, CV={cv_percent:.1f}%")

    axes[0][col].imshow(cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB))
    axes[0][col].set_title(name); axes[0][col].axis("off")
    # Intensity profile (horizontal mid-line)
    axes[1][col].plot(gray[h//2, :], color="steelblue")
    axes[1][col].set_title(f"Horizontal profile – {name}")
    axes[1][col].set_xlabel("Column (px)"); axes[1][col].set_ylabel("Intensity")

plt.suptitle("Task 7-c  Illumination Simulation & Uniformity", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task7c_illumination.png", dpi=120)
plt.close()

illum_df = pd.DataFrame(illum_rows)
illum_df.to_csv("outputs/task7c_illumination_metrics.csv", index=False)
print("Saved → outputs/task7c_illumination.png, task7c_illumination_metrics.csv")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 7-d  Select exposure and lighting; save JSON
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 7-d  Selected Parameters")
print("=" * 60)

# Best exposure: shortest time with blur < 0.5 px → 0.1 ms
selected_exposure_ms = 0.1

# Best lighting: lowest CV (most uniform)
best_light = illum_df.loc[illum_df["CV_pct"].idxmin(), "Light"]

selected_params = {
    "exposure_ms":        selected_exposure_ms,
    "exposure_justification": "0.1 ms gives blur < 0.5 px at 0.5 m/s belt speed",
    "lighting_type":       best_light,
    "lighting_cv_pct":     float(illum_df.loc[illum_df["Light"]==best_light,"CV_pct"]),
    "lighting_justification": f"{best_light} provides lowest spatial variation (CV)",
    "belt_speed_ms":       BELT_SPEED_MS,
    "fov_mm":              FOV_MM,
}

with open("outputs/task7d_selected_params.json", "w") as f:
    json.dump(selected_params, f, indent=2)
print(f"  Selected exposure : {selected_exposure_ms} ms")
print(f"  Selected lighting : {best_light}  (CV={selected_params['lighting_cv_pct']:.2f}%)")
print("  Saved → outputs/task7d_selected_params.json")
print("\nTask 7 COMPLETE")
