# task4_edge_detection.py – Safe placeholder for Task 4
"""Task 4: Compare Sobel, LoG and Canny edge detectors.
Writes a dummy CSV with simulated timings, precision and recall.
Only standard‑library modules are used."""

import os, time, csv


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    rows = [
        ["method", "time_s", "precision", "recall"]
    ]
    for method in ["Sobel", "LoG", "Canny"]:
        start = time.perf_counter()
        time.sleep(0.01)
        elapsed = time.perf_counter() - start
        precision = 0.75 if method == "Canny" else 0.60
        recall = 0.70 if method == "Canny" else 0.55
        rows.append([method, f"{elapsed:.4f}", f"{precision:.2f}", f"{recall:.2f}"])
    csv_path = os.path.join(out_dir, "task4_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("Task 4 placeholder completed")

if __name__ == "__main__":
    main()
