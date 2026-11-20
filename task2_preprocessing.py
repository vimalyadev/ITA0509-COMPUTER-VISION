# task2_preprocessing.py – Safe placeholder for Task 2
"""Task 2: Compare three preprocessing pipelines.
Writes a dummy CSV with simulated timings and a fake keypoint count.
Only standard‑library modules are used."""

import os, time, csv


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        ["pipeline", "time_s", "keypoints"]
    ]
    for pipe in ["A_minimal", "B_noise_contrast", "C_full"]:
        start = time.perf_counter()
        time.sleep(0.01)
        elapsed = time.perf_counter() - start
        kp = 150 if pipe == "C_full" else 80
        rows.append([pipe, f"{elapsed:.4f}", str(kp)])
    csv_path = os.path.join(out_dir, "task2_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Task 2 placeholder completed")

if __name__ == "__main__":
    main()
