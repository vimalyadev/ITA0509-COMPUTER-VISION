# generate_report.py – Simple PDF report using fpdf2
"""Create a minimal PDF summarising Assignment 2.
For each task we check if the corresponding CSV metric file exists in the
outputs folder and write a line with the task name and status.
This implementation uses the pure‑Python fpdf2 library to avoid native crashes.
"""

import os
import pandas as pd
from fpdf import FPDF


def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    base_dir = os.path.dirname(__file__)
    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "Assignment2_Report.pdf")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Smart Document Scanner – Assignment 2 Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", size=12)

    tasks = [
        ("Task 1 – Enhancement", "task1_metrics.csv"),
        ("Task 2 – Histogram Equalization", "task2_metrics.csv"),
        ("Task 3 – Noise & Denoise", "task3_metrics.csv"),
        ("Task 4 – Sharpening", "task4_metrics.csv"),
        ("Task 5 – Low‑Pass Filtering", "task5_metrics.csv"),
        ("Task 6 – High‑Pass Filtering", "task6_metrics.csv"),
        ("Task 7 – Edge Detection", "task7_metrics.csv"),
        ("Task 8 – Thresholding", "task8_metrics.csv"),
        ("Task 9 – Segmentation", "task9_metrics.csv"),
        ("Task 10 – Integrated Summary", "summary_metrics.csv"),
    ]

    for title, csv_name in tasks:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, title, ln=True)
        csv_path = os.path.join(out_dir, csv_name)
        df = load_csv(csv_path)
        pdf.set_font("Helvetica", size=11)
        if df is not None and not df.empty:
            pdf.cell(0, 6, f"Rows: {len(df)}  Columns: {', '.join(df.columns.tolist()[:4])}", ln=True)
        else:
            pdf.set_font("Helvetica", "I", 11)
            pdf.cell(0, 6, "No data – task not executed or CSV missing.", ln=True)
        pdf.ln(2)
    pdf.output(pdf_path)
    print(f"Report written to {pdf_path}")

if __name__ == "__main__":
    main()
