"""
run_all.py – Execute Assignment 2 tasks in order
================================================
Run from the Assignment_2 directory:
    python run_all.py
"""

import subprocess, sys, os, time

TASKS = [
    "generate_report.py",
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
print("="*70)
print(" Smart Document Scanner – Assignment 2")
print("="*70)
print(f" Working directory: {os.getcwd()}")
print()

results = {}
for script in TASKS:
    print(f"\n{'─'*70}")
    print(f" ► Running {script} …")
    print(f"{'─'*70}")
    t0 = time.perf_counter()
    ret = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.perf_counter() - t0
    status = "✓ OK" if ret.returncode == 0 else f"✗ ERROR (code {ret.returncode})"
    results[script] = (ret.returncode, elapsed)
    print(f"\n  → {status}  ({elapsed:.1f}s)")

print(f"\n{'='*70}")
print(" Summary")
print(f"{'='*70}")
for script, (code, elapsed) in results.items():
    ok = "✓" if code == 0 else "✗"
    print(f"  {ok}  {script:<35}  {elapsed:.1f}s")

all_ok = all(c == 0 for c, _ in results.values())
print(f"\n{'All tasks completed successfully!' if all_ok else 'Some tasks failed – see output above.'}")
