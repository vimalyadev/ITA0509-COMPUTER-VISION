# generate_report_static.py – Minimal PDF report
"""Create a simple PDF that states the assignment tasks and that no data was generated.
All tasks are listed and marked as *not executed*.
Uses fpdf2 (pure Python) which works on Windows.
"""

import os
from fpdf import FPDF


def main():
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Assignment2_Report.pdf")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Smart Document Scanner – Assignment 2 Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)
    tasks = [
        "Task 1 – Enhancement",
        "Task 2 – Histogram Equalization",
        "Task 3 – Noise & Denoise",
        "Task 4 – Sharpening",
        "Task 5 – Low‑Pass Filtering",
        "Task 6 – High‑Pass Filtering",
        "Task 7 – Edge Detection",
        "Task 8 – Thresholding",
        "Task 9 – Segmentation",
        "Task 10 – Integrated Summary",
    ]
    for t in tasks:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, t, ln=True)
        pdf.set_font("Helvetica", "I", 11)
        pdf.cell(0, 6, "No data – task not executed.", ln=True)
        pdf.ln(2)
    pdf.output(pdf_path)
    print(f"Report written to {pdf_path}")

if __name__ == "__main__":
    main()
