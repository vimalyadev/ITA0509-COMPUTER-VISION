# Assignment 2 – Smart Document Scanner

**Goal**: Apply image‑enhancement and segmentation techniques to digitised document pages (photos taken with a phone).  The pipeline must handle shadows, low contrast, stains, skew and clutter, then isolate text/diagrams for OCR.

The repository contains:
- `generate_dataset.py` – creates a synthetic dataset of 30 document images with corresponding **text masks** (binary PNG where the foreground text = 255).  The images mimic realistic phone‑camera artefacts (shadow gradient, blur, rotation, stain blobs).
- `task1_enhancement.py` – linear stretching, gamma and logarithmic transforms; histograms; contrast/entropy/recall measurement.
- `task2_histogram.py` – global histogram equalisation and CLAHE; histograms & CDFs; SSIM/recall.
- `task3_noise.py` – add Gaussian, salt‑pepper, speckle noise; denoising filters; PSNR/SSIM.
- `task4_sharpen.py` – unsharp mask, Laplacian, high‑boost; intensity profiles; edge‑sharpness.
- `task5_lowpass_fft.py` – Fourier magnitude, ideal/Butterworth/Gaussian low‑pass; PSNR/SSIM/running‑time.
- `task6_highpass_fft.py` – ideal/Butterworth/Gaussian high‑pass; edge contrast & recall.
- `task7_edge_detection.py` – point/line kernels, Sobel, Canny; precision/recall/F1.
- `task8_thresholding.py` – fixed, Otsu, adaptive Gaussian; Dice & IoU.
- `task9_segmentation.py` – connected components, watershed, morph‑linking; IoU & segmentation errors.
- `task10_integrated.py` – end‑to‑end scanner function, robustness test, CSV metrics.
- `run_all.py` – executes the ten tasks in order.
- `generate_report.py` – builds a PDF summary of all results.

All scripts write their artefacts to the `outputs/` folder.  See the individual task docs for usage.

---
**Dataset**
The synthetic data are generated on‑the‑fly; no external download is required.  Each image is `800×800` px, black‑on‑white text (OpenCV Hershey fonts) with random rotations (`‑15°…15°`), a soft shadow gradient, Gaussian blur (optional), and random ink‑stain blobs.

---
**Running the whole pipeline**
```powershell
cd "c:\computer vision assignmnet\Assignment_2"
python run_all.py
```
The final PDF will appear at `outputs/Assignment2_Report.pdf`.
