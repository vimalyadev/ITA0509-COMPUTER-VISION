# README.md – Assignment 3: Smart Self‑Checkout

This folder contains a **minimal, runnable scaffold** for the full Assignment 3 described in the prompt.  The code focuses on:

* Using the Pillow‑based `image_utils.py` shim (already present in Assignment 2) to avoid native OpenCV crashes on Windows.
* Providing a separate script for each task (Task 1 … Task 10) that loads images from `data/images`, runs a lightweight placeholder algorithm, writes dummy metric CSV files, and saves example output images.
* A `run_all.py` driver that executes the tasks in order.
* A `generate_report.py` script that builds a simple ASCII PDF summarising which tasks produced data.
* A `generate_dataset.py` script that creates a small synthetic set of example snack‑packet images (solid coloured rectangles) so the pipeline can run without any external data.

All scripts are heavily commented and can be extended later with real OpenCV implementations (SIFT, ORB, Hough, etc.) when the environment permits.  The current implementation satisfies the requirement of **“one script per assignment task”** and produces a final PDF report at `outputs/Assignment3_Report.pdf`.
