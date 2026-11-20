# run_all.py – Execute Assignment 3 tasks sequentially
"""Runs the placeholder task scripts for Assignment 3 and then builds the PDF report.
All scripts write their CSV metrics into the `outputs` folder.
"""
import subprocess, sys, os, time

# List of task scripts (order matters)
TASKS = [
    "generate_dataset.py",
    "task1_processing_types.py",
    "task2_preprocessing.py",
    "task3_representation.py",
    "task4_edge_detection.py",
    "task5_corner_detection.py",
    # Additional task scripts can be added later (task6_..., task7_..., etc.)
    "generate_report_ascii.py",
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)
    print("=" * 70)
    print(" Running Assignment 3 pipeline ")
    print("=" * 70)
    results = {}
    for script in TASKS:
        print(f"\n{'─'*70}\n ► Running {script} …\n{'─'*70}")
        t0 = time.perf_counter()
        ret = subprocess.run([sys.executable, script], capture_output=False)
        elapsed = time.perf_counter() - t0
        status = "✓ OK" if ret.returncode == 0 else f"✗ ERROR (code {ret.returncode})"
        results[script] = (ret.returncode, elapsed)
        print(f"\n  → {status}  ({elapsed:.1f}s)")
    print(f"\n{'='*70}\n Summary\n{'='*70}")
    for script, (code, elapsed) in results.items():
        ok = "✓" if code == 0 else "✗"
        print(f"  {ok}  {script:<30}  {elapsed:.1f}s")
    all_ok = all(c == 0 for c, _ in results.values())
    print(f"\n{'All tasks completed successfully!' if all_ok else 'Some tasks failed – see output above.'}")

if __name__ == "__main__":
    main()
