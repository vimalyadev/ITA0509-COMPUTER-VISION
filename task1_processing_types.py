# task1_processing_types.py – Safe placeholder for Task 1
"""Task 1: Compare low‑level, low+mid, and low+mid+high pipelines.
Writes a dummy CSV with simulated timings and accuracies using only the
standard library (no pandas, no OpenCV)."""

import os, time, csv


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        ["pipeline", "time_s", "accuracy"]
    ]
    for typ in ["low", "low_mid", "low_mid_high"]:
        start = time.perf_counter()
        time.sleep(0.01)  # simulate work
        elapsed = time.perf_counter() - start
        accuracy = 0.90 if typ == "low_mid_high" else 0.60
        rows.append([typ, f"{elapsed:.4f}", f"{accuracy:.2f}"])
    csv_path = os.path.join(out_dir, "task1_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Task 1 placeholder completed")

if __name__ == "__main__":
    main()
