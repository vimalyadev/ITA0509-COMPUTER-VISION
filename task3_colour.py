"""
task3_colour.py – Colour Representation for Ripeness
======================================================
Task 3 of Assignment 1: GreenHarvest Mango Grading
- a) Convert fruit set to BGR, HSV, Lab; save labelled versions
- b) Extract masked channel distributions for green/turning/ripe
- c) Train classifiers per representation; report accuracy & CM
- d) Brightness perturbation; accuracy loss; select best representation
"""

import cv2, numpy as np, pandas as pd, os, json
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

os.makedirs("outputs", exist_ok=True)

with open("data/metadata.json") as f:
    META = json.load(f)

CLASSES = ["green", "turning", "ripe"]

def load_all():
    imgs, labels = [], []
    for m in META:
        p = os.path.join("data/images", m["file"])
        img = cv2.imread(p)
        if img is not None:
            imgs.append(img)
            labels.append(m["label"])
    return imgs, labels

IMAGES, LABELS = load_all()
N = len(IMAGES)
print(f"Loaded {N} images")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-a  Convert and save labelled colour-space images
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-a  Colour Space Conversions")
print("="*60)

REPR_SAVE_DIR = "outputs/task3a_colourspaces"
os.makedirs(REPR_SAVE_DIR, exist_ok=True)

for m, img in zip(META[:6], IMAGES[:6]):   # save first 6 for each space
    base = f"id{m['id']:03d}_{m['label']}"
    # BGR (original)
    bgr_lbl = img.copy()
    cv2.putText(bgr_lbl, f"BGR {m['label']}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imwrite(f"{REPR_SAVE_DIR}/{base}_BGR.png", bgr_lbl)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_lbl = hsv.copy()
    cv2.putText(hsv_lbl, f"HSV {m['label']}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imwrite(f"{REPR_SAVE_DIR}/{base}_HSV.png", hsv_lbl)

    # Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    lab_lbl = lab.copy()
    cv2.putText(lab_lbl, f"Lab {m['label']}", (5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.imwrite(f"{REPR_SAVE_DIR}/{base}_Lab.png", lab_lbl)

print(f"Saved labelled images → {REPR_SAVE_DIR}/")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-b  Channel distributions per ripeness class
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-b  Channel Distribution Plots")
print("="*60)

def get_peel_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Return binary mask of mango peel pixels."""
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 30, 50), (180, 255, 255))
    return mask

def extract_channel_stats(images, labels, space: str) -> pd.DataFrame:
    rows = []
    for img, lbl in zip(images, labels):
        mask = get_peel_mask(img)
        if space == "BGR":
            converted = img
            ch_names = ["B","G","R"]
        elif space == "HSV":
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ch_names = ["H","S","V"]
        else:  # Lab
            converted = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
            ch_names = ["L","a","b"]
        for ci, cn in enumerate(ch_names):
            vals = converted[:,:,ci][mask > 0]
            if len(vals):
                rows.append({"label": lbl, "channel": cn,
                             "mean": float(vals.mean()), "std": float(vals.std())})
    return pd.DataFrame(rows)

colour_spaces = ["BGR", "HSV", "Lab"]
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

for row_idx, space in enumerate(colour_spaces):
    ch_names = {"BGR":["B","G","R"], "HSV":["H","S","V"], "Lab":["L","a","b"]}[space]
    for col_idx, ch in enumerate(ch_names):
        ax = axes[row_idx][col_idx]
        for lbl, color in zip(CLASSES, ["green","orange","red"]):
            idxs = [i for i,l in enumerate(LABELS) if l==lbl]
            vals_all = []
            for i in idxs:
                img  = IMAGES[i]
                mask = get_peel_mask(img)
                if space == "BGR":
                    conv = img
                elif space == "HSV":
                    conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                else:
                    conv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                px = conv[:,:,col_idx][mask>0]
                vals_all.extend(px.tolist())
            if vals_all:
                ax.hist(vals_all, bins=50, alpha=0.6, label=lbl, color=color, density=True)
        ax.set_title(f"{space} – {ch}")
        ax.set_xlabel("Channel value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

plt.suptitle("Task 3-b  Channel Distributions per Ripeness Label", fontsize=13)
plt.tight_layout()
plt.savefig("outputs/task3b_channel_distributions.png", dpi=120)
plt.close()
print("Saved → outputs/task3b_channel_distributions.png")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-c  Classifier per colour space
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-c  SVM Classifiers per Colour Space")
print("="*60)

le = LabelEncoder()
le.fit(CLASSES)
y  = le.transform(LABELS)

def build_features(images, space):
    X = []
    for img in images:
        mask = get_peel_mask(img)
        if space == "BGR":
            conv = img
        elif space == "HSV":
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        else:
            conv = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        feats = []
        for ci in range(3):
            vals = conv[:,:,ci][mask>0]
            if len(vals):
                feats += [vals.mean(), vals.std()]
            else:
                feats += [0.0, 0.0]
        X.append(feats)
    return np.array(X)

results = []
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, space in zip(axes, colour_spaces):
    X = build_features(IMAGES, space)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc  = accuracy_score(y_te, y_pred)
    cm   = confusion_matrix(y_te, y_pred)
    print(f"\n  Space={space}  Accuracy={acc:.3f}")
    print(classification_report(y_te, y_pred, target_names=CLASSES))
    results.append({"Space": space, "Accuracy": acc})

    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{space} SVM\nAcc={acc:.2f}")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(CLASSES, rotation=45); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i,j], ha="center", va="center",
                    color="white" if cm[i,j]>cm.max()//2 else "black")

plt.suptitle("Task 3-c  Confusion Matrices per Colour Space", fontsize=12)
plt.tight_layout()
plt.savefig("outputs/task3c_classifier_cm.png", dpi=120)
plt.close()
print("\nSaved → outputs/task3c_classifier_cm.png")

# ═══════════════════════════════════════════════════════════════════════════
# TASK 3-d  Brightness perturbation
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("TASK 3-d  Brightness Perturbation & Representation Stability")
print("="*60)

def perturb_brightness(img_bgr: np.ndarray, delta: int) -> np.ndarray:
    """Add delta to V-channel (HSV) to simulate brightness change."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:,:,2] = np.clip(hsv[:,:,2] + delta, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

delta = 60   # strong brightness perturbation
IMAGES_BRIGHT = [perturb_brightness(img, delta) for img in IMAGES]

acc_rows = []
for space in colour_spaces:
    # Normal
    X_norm  = build_features(IMAGES, space)
    X_tr, X_te, y_tr, y_te = train_test_split(X_norm, y, test_size=0.3,
                                               random_state=42, stratify=y)
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_tr, y_tr)
    acc_norm  = accuracy_score(y_te, clf.predict(X_te))

    # Bright perturbed test set
    X_bright = build_features(IMAGES_BRIGHT, space)
    _, X_te_b, _, _ = train_test_split(X_bright, y, test_size=0.3,
                                        random_state=42, stratify=y)
    acc_bright = accuracy_score(y_te, clf.predict(X_te_b))

    loss = acc_norm - acc_bright
    acc_rows.append({"Space": space, "Acc_Normal": round(acc_norm,3),
                     "Acc_Bright": round(acc_bright,3), "Acc_Loss": round(loss,3)})
    print(f"  {space}: normal={acc_norm:.3f}  bright={acc_bright:.3f}  loss={loss:.3f}")

acc_df = pd.DataFrame(acc_rows)
acc_df.to_csv("outputs/task3d_brightness_robustness.csv", index=False)

# Bar chart
x = np.arange(len(colour_spaces))
w = 0.3
fig, ax = plt.subplots(figsize=(8,4))
ax.bar(x-w/2, acc_df["Acc_Normal"], w, label="Normal", color="steelblue")
ax.bar(x+w/2, acc_df["Acc_Bright"], w, label="Bright (+60)", color="darkorange")
ax.set_xticks(x); ax.set_xticklabels(colour_spaces)
ax.set_ylabel("Accuracy"); ax.set_ylim(0,1)
ax.legend(); ax.set_title("Task 3-d  Brightness Robustness per Colour Space")
plt.tight_layout()
plt.savefig("outputs/task3d_brightness_robustness.png", dpi=120)
plt.close()

best = acc_df.loc[acc_df["Acc_Loss"].idxmin(), "Space"]
print(f"\nBest representation (lowest accuracy loss): {best}")
print(f"Selection evidence saved → outputs/task3d_brightness_robustness.csv")
print("\nTask 3 COMPLETE")
