"""
generate_report.py – Create a PDF report of all 10 tasks
=========================================================
Uses ReportLab to assemble a professional assignment report.
Run after run_all.py:
    python generate_report.py
"""

import os, json, csv
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── Styles ───────────────────────────────────────────────────────────────
styles = getSampleStyleSheet()
style_title   = ParagraphStyle("Title2",   parent=styles["Title"],
                                fontSize=20, spaceAfter=12, alignment=TA_CENTER,
                                textColor=colors.HexColor("#1a4a1a"))
style_h1      = ParagraphStyle("H1",       parent=styles["Heading1"],
                                fontSize=14, spaceAfter=6,
                                textColor=colors.HexColor("#1a4a1a"))
style_h2      = ParagraphStyle("H2",       parent=styles["Heading2"],
                                fontSize=12, spaceAfter=4,
                                textColor=colors.HexColor("#2d7a2d"))
style_body    = ParagraphStyle("Body",     parent=styles["Normal"],
                                fontSize=10, spaceAfter=4, leading=14,
                                alignment=TA_JUSTIFY)
style_code    = ParagraphStyle("Code",     parent=styles["Code"],
                                fontSize=8,  spaceAfter=4, leading=11,
                                fontName="Courier")
style_caption = ParagraphStyle("Caption",  parent=styles["Normal"],
                                fontSize=8,  spaceAfter=6, alignment=TA_CENTER,
                                textColor=colors.grey)
style_tbl_hdr = ParagraphStyle("TblHdr",  parent=styles["Normal"],
                                fontSize=9,  fontName="Helvetica-Bold",
                                textColor=colors.white)

PAGE_W, PAGE_H = A4
IMG_W_FULL = PAGE_W - 4*cm
IMG_W_HALF = IMG_W_FULL / 2 - 0.5*cm

def add_image(path, width=None, caption=None):
    """Return flowable list for an image with optional caption."""
    if not os.path.exists(path):
        return [Paragraph(f"[Image not found: {path}]", style_caption)]
    w = width or IMG_W_FULL
    items = [RLImage(path, width=w, height=w*0.6)]
    if caption:
        items.append(Paragraph(caption, style_caption))
    return items

def tbl(data, col_widths=None):
    """Create a styled ReportLab Table."""
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("BACKGROUND",   (0,0),(-1,0),  colors.HexColor("#1a4a1a")),
        ("TEXTCOLOR",    (0,0),(-1,0),  colors.white),
        ("FONTNAME",     (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0,0),(-1,-1), 8),
        ("GRID",         (0,0),(-1,-1), 0.5, colors.grey),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white, colors.HexColor("#e8f4e8")]),
        ("VALIGN",       (0,0),(-1,-1), "MIDDLE"),
        ("LEFTPADDING",  (0,0),(-1,-1), 4),
        ("RIGHTPADDING", (0,0),(-1,-1), 4),
    ]))
    return t

def csv_to_table(csv_path, max_rows=10):
    """Read CSV and return a Table flowable."""
    if not os.path.exists(csv_path):
        return Paragraph(f"[CSV not found: {csv_path}]", style_caption)
    rows = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > max_rows: break
            rows.append(row)
    if not rows: return Paragraph("[Empty CSV]", style_caption)
    return tbl(rows)

# ═══════════════════════════════════════════════════════════════════════════
# Document assembly
# ═══════════════════════════════════════════════════════════════════════════
doc = SimpleDocTemplate(
    "outputs/Assignment1_Report.pdf",
    pagesize=A4,
    leftMargin=2*cm, rightMargin=2*cm,
    topMargin=2.5*cm, bottomMargin=2*cm,
)

story = []

# ── Cover page ───────────────────────────────────────────────────────────
story += [
    Spacer(1, 2*cm),
    Paragraph("COMPUTER VISION – ASSIGNMENT 1", style_title),
    Spacer(1, 0.3*cm),
    HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a4a1a")),
    Spacer(1, 0.3*cm),
    Paragraph("Apply Image Formation and Digital Image Fundamentals<br/>to a Mango Grading Line", style_title),
    Spacer(1, 1.5*cm),
]

cover_data = [
    ["Field",          "Value"],
    ["Student",        "GreenHarvest Vision Engineer"],
    ["Institution",    "Computer Vision Course – 2026"],
    ["Total Marks",    "50 (10 Tasks × 5 Marks)"],
    ["Dataset",        "30 synthetic mango images (parameterised)"],
    ["Language",       "Python 3 (OpenCV, NumPy, scikit-learn, matplotlib)"],
    ["Report Date",    "2026-11-19"],
]
story.append(tbl(cover_data, col_widths=[5*cm, 11*cm]))
story += [Spacer(1, 1*cm), PageBreak()]

# ── Scenario overview ────────────────────────────────────────────────────
story += [
    Paragraph("Scenario Overview", style_h1),
    Paragraph(
        "GreenHarvest Cooperative moves mangoes at <b>0.5 m/s</b> beneath a camera. "
        "The vision system must: measure fruit size, classify ripeness from peel colour, "
        "and detect defects at least <b>2 mm wide</b> across a <b>400 mm field of view</b>. "
        "The decision limit is <b>1.2 s per fruit</b>. "
        "A dataset of 30 synthetic mango images (10 per ripeness label, half with defects) "
        "was generated programmatically and used throughout all tasks.",
        style_body,
    ),
    Spacer(1, 0.5*cm),
    HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2d7a2d")),
    PageBreak(),
]

# ── Helper to add a task section ─────────────────────────────────────────
def task_section(num, title, description, subtasks, images, csv_paths):
    items = []
    items += [
        Paragraph(f"Task {num} – {title}  [{5} Marks]", style_h1),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#2d7a2d")),
        Spacer(1, 0.2*cm),
        Paragraph(description, style_body),
        Spacer(1, 0.3*cm),
    ]
    for sub_label, sub_text in subtasks:
        items += [
            Paragraph(f"<b>{sub_label}</b>  {sub_text}", style_body),
        ]
    items.append(Spacer(1, 0.4*cm))

    # Images
    for img_path, caption in images:
        items += add_image(img_path, caption=caption)
        items.append(Spacer(1, 0.3*cm))

    # CSV tables
    for csv_path, label in csv_paths:
        if os.path.exists(csv_path):
            items += [
                Paragraph(f"<b>{label}</b>", style_body),
                csv_to_table(csv_path),
                Spacer(1, 0.3*cm),
            ]

    items.append(PageBreak())
    return items

# ── Task 1 ───────────────────────────────────────────────────────────────
story += task_section(
    1, "Vision Systems and Processing Levels",
    "A three-level pipeline (low, mid, high) was implemented. Three grading approaches "
    "(simulated manual, image-processing-assisted, rule-based) were compared using Cohen's κ "
    "and confusion matrices. Throughput was measured against the 8 fruits/s target.",
    [
        ("a)", "Three-stage pipeline with printed representations for each stage."),
        ("b)", "Three grading approaches with decisions saved to CSV."),
        ("c)", "Confusion matrices and Cohen's κ computed with scikit-learn."),
        ("d)", "Latency measured with time.perf_counter; fps vs 8 fps target."),
    ],
    [
        ("outputs/task1c_confusion_matrices.png", "Fig 1.1  Confusion Matrices – Three Grading Approaches"),
        ("outputs/task1d_throughput.png",         "Fig 1.2  Throughput Comparison vs 8 fps Target"),
    ],
    [
        ("outputs/task1_grading_decisions.csv", "Table 1.1  Grading Decisions (first 10 rows)"),
        ("outputs/task1d_latency.csv",          "Table 1.2  Latency and Throughput Metrics"),
    ],
)

# ── Task 2 ───────────────────────────────────────────────────────────────
story += task_section(
    2, "Image Arrays, Coordinates, and Storage",
    "A sample mango image was loaded and its array properties examined. A coordinate probe "
    "mapped display (x,y) to NumPy [row,col]. Storage was calculated from dimensions and "
    "bit depth and verified. cv2.add (saturating) was contrasted with NumPy uint8 overflow.",
    [
        ("a)", "shape, dtype, channel count, ndarray.nbytes reported."),
        ("b)", "Probe: (x,y) → [row,col] with BGR and RGB pixel values."),
        ("c)", "Storage formula verified against ndarray.nbytes."),
        ("d)", "cv2.add vs NumPy addition on bright pixels; overflow pixels reported."),
    ],
    [
        ("outputs/task2b_probed_image.png",         "Fig 2.1  Probed Pixel Locations"),
        ("outputs/task2d_comparison_plot.png",       "Fig 2.2  Saturating vs Wrapping Addition"),
    ],
    [
        ("outputs/task2b_coordinate_probe.csv", "Table 2.1  Coordinate Probe Results"),
    ],
)

# ── Task 3 ───────────────────────────────────────────────────────────────
story += task_section(
    3, "Colour Representation for Ripeness",
    "The mango dataset was converted to BGR, HSV, and CIE Lab colour spaces. "
    "Channel distributions were plotted per ripeness class. Equivalent SVM classifiers "
    "were trained for each space and evaluated. Brightness perturbation identified the "
    "most robust representation.",
    [
        ("a)", "BGR, HSV, Lab images saved with labels."),
        ("b)", "Channel histograms plotted for green/turning/ripe classes."),
        ("c)", "SVM classifiers trained per space; accuracy and CM reported."),
        ("d)", "Brightness perturbation (+60 V-channel); accuracy loss compared."),
    ],
    [
        ("outputs/task3b_channel_distributions.png", "Fig 3.1  Channel Distributions per Ripeness"),
        ("outputs/task3c_classifier_cm.png",          "Fig 3.2  Confusion Matrices per Colour Space"),
        ("outputs/task3d_brightness_robustness.png",  "Fig 3.3  Brightness Robustness Comparison"),
    ],
    [
        ("outputs/task3d_brightness_robustness.csv", "Table 3.1  Brightness Robustness Metrics"),
    ],
)

# ── Task 4 ───────────────────────────────────────────────────────────────
story += task_section(
    4, "Spatial Sampling and Defect Visibility",
    "Mango images were resized to three resolutions (320×240, 160×120, 80×60) using "
    "INTER_NEAREST and INTER_AREA. mm/pixel and pixels-per-2mm were calculated. "
    "Defect recall was evaluated at each scale.",
    [
        ("a)", "Resize with INTER_NEAREST and INTER_AREA; images saved."),
        ("b)", "mm/pixel and pixels spanning a 2 mm defect for each resolution."),
        ("c)", "Defect recall and apparent area vs annotated masks."),
        ("d)", "Resolution chosen providing ≥ 4 px across the smallest defect."),
    ],
    [
        ("outputs/task4d_defect_crops.png", "Fig 4.1  Defect Crops at Different Resolutions"),
    ],
    [
        ("outputs/task4b_scale_calculations.csv", "Table 4.1  Scale Calculations"),
        ("outputs/task4c_defect_recall.csv",       "Table 4.2  Defect Recall per Resolution"),
    ],
)

# ── Task 5 ───────────────────────────────────────────────────────────────
story += task_section(
    5, "Intensity Quantization",
    "Images were quantized to 8-, 6-, and 4-bit depth. Local contrast between healthy "
    "peel and defect regions was measured. Precision and recall of defect detection were "
    "computed at each bit depth and the minimum adequate depth selected.",
    [
        ("a)", "8-bit, 6-bit, 4-bit images created with NumPy; saved."),
        ("b)", "Contrast (healthy peel vs defect) measured before and after quantization."),
        ("c)", "Threshold applied; precision and recall computed vs GT masks."),
        ("d)", "Bit-depth vs storage, contrast, recall plotted; minimum adequate selected."),
    ],
    [
        ("outputs/task5a_quantization_comparison.png", "Fig 5.1  Quantization Comparison"),
        ("outputs/task5d_bitdepth_tradeoff.png",        "Fig 5.2  Bit-Depth Trade-off"),
    ],
    [
        ("outputs/task5b_contrast.csv",     "Table 5.1  Contrast Metrics"),
        ("outputs/task5c_defect_metrics.csv","Table 5.2  Defect Precision & Recall"),
    ],
)

# ── Task 6 ───────────────────────────────────────────────────────────────
story += task_section(
    6, "Image Formation and Lens Geometry",
    "Pinhole projection was implemented from focal length, object distance, and mango width. "
    "Synthetic checkerboard calibration produced a camera matrix and RMS reprojection error. "
    "Two configurations were compared and the geometry was plotted to justify placement.",
    [
        ("a)", "Pinhole: f = (sensor/FOV)×d; magnification and image size computed."),
        ("b)", "Camera matrix and RMS reprojection error from calibration."),
        ("c)", "Two lens/distance configurations compared for mm/px and size error."),
        ("d)", "Camera–object–FOV geometry plotted; placement justified."),
    ],
    [
        ("outputs/task6d_geometry_plot.png", "Fig 6.1  Camera Geometry – Object Plane and FOV"),
    ],
    [
        ("outputs/task6c_config_comparison.csv", "Table 6.1  Lens Configuration Comparison"),
    ],
)

# ── Task 7 ───────────────────────────────────────────────────────────────
story += task_section(
    7, "Illumination and Exposure",
    "Motion blur was calculated for three exposures at 0.5 m/s. A gamma-based exposure "
    "sweep was performed and illumination uniformity was measured for diffuse, side-light, "
    "and ring-light simulations. Optimal exposure and lighting were selected.",
    [
        ("a)", "Blur computed at 0.1, 1.0, 4.0 ms; visualised with convolution kernels."),
        ("b)", "Gamma sweep (0.5–2.0) simulating different exposures; images saved."),
        ("c)", "Coefficient of variation for each lighting type measured."),
        ("d)", "0.1 ms selected (blur < 0.5 px); lighting chosen by lowest CV."),
    ],
    [
        ("outputs/task7a_motion_blur_simulation.png", "Fig 7.1  Motion Blur Simulation"),
        ("outputs/task7b_exposure_sweep.png",          "Fig 7.2  Exposure Sweep"),
        ("outputs/task7c_illumination.png",            "Fig 7.3  Illumination Simulations"),
    ],
    [
        ("outputs/task7a_motion_blur.csv",         "Table 7.1  Motion Blur Metrics"),
        ("outputs/task7c_illumination_metrics.csv", "Table 7.2  Illumination Uniformity"),
    ],
)

# ── Task 8 ───────────────────────────────────────────────────────────────
story += task_section(
    8, "Connectivity and Defect Measurements",
    "Binary masks with isolated, touching, and diagonal defects were created and labelled "
    "with 4- and 8-connectivity. Component areas were compared with ground truth. "
    "Euclidean, city-block, and chessboard distance transforms were applied and a 4 mm "
    "rejection rule implemented.",
    [
        ("a)", "Three defect topology types created; labelled with both connectivities."),
        ("b)", "Component counts and areas reported vs GT annotations."),
        ("c)", "Three distance metrics applied with mm calibration."),
        ("d)", "4 mm rejection rule evaluated; FP/FN counts reported."),
    ],
    [
        ("outputs/task8c_distance_transforms.png", "Fig 8.1  Distance Transforms"),
        ("outputs/task8d_labels_dt.png",           "Fig 8.2  Label Map and Euclidean DT"),
    ],
    [
        ("outputs/task8b_component_areas.csv",    "Table 8.1  Component Areas"),
        ("outputs/task8d_rejection_results.csv",  "Table 8.2  4 mm Rejection Results"),
    ],
)

# ── Task 9 ───────────────────────────────────────────────────────────────
story += task_section(
    9, "Image Acquisition Arrangements",
    "Area-scan blur, rolling-shutter row displacement, and line-scan stacking were "
    "simulated. The required line rate at 0.5 m/s was calculated. A 100 Hz illumination "
    "modulation was applied and intensity variation compared across arrangements. "
    "Line-scan was selected.",
    [
        ("a)", "Area-scan blur, rolling-shutter, line-scan simulations saved."),
        ("b)", "Line rate = H × v / FOV = 300 lines/s calculated and printed."),
        ("c)", "100 Hz modulation applied; row/frame intensity CV computed."),
        ("d)", "Comparison table: blur, skew, flicker, data rate, runtime."),
    ],
    [
        ("outputs/task9a_simulations.png", "Fig 9.1  Acquisition Arrangement Simulations"),
    ],
    [
        ("outputs/task9c_modulation_variation.csv", "Table 9.1  100 Hz Modulation Intensity CV"),
        ("outputs/task9d_comparison.csv",           "Table 9.2  Acquisition Arrangement Comparison"),
    ],
)

# ── Task 10 ──────────────────────────────────────────────────────────────
story += task_section(
    10, "Integrated Front End and Design Decision",
    "All acquisition, colour, segmentation, and defect measurement steps were combined into "
    "a single callable function. Three configurations were benchmarked on the full test set "
    "and the least costly configuration meeting all constraints was selected.",
    [
        ("a)", "grade_mango() callable defined integrating all pipeline stages."),
        ("b)", "Full test set evaluated; CSV with ID, width, ripeness, defects, latency, grade."),
        ("c)", "Latency and defect recall plotted for three configurations."),
        ("d)", "Final JSON config and design report generated."),
    ],
    [
        ("outputs/task10c_config_comparison.png", "Fig 10.1  Configuration Comparison"),
    ],
    [
        ("outputs/task10b_grading_results.csv", "Table 10.1  Grading Results (first 10 rows)"),
        ("outputs/task10c_config_summary.csv",  "Table 10.2  Configuration Summary"),
    ],
)

# ── Final page – Selected design ─────────────────────────────────────────
if os.path.exists("outputs/task10d_final_config.json"):
    with open("outputs/task10d_final_config.json") as f:
        cfg = json.load(f)

    story += [
        Paragraph("Selected Design Summary", style_h1),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a4a1a")),
        Spacer(1, 0.4*cm),
    ]
    design_data = [
        ["Parameter", "Value"],
        ["Configuration",       cfg.get("selected_configuration","–")],
        ["Resolution",          cfg.get("resolution","–")],
        ["Bit depth",           str(cfg.get("bits","–")) + " bits"],
        ["Exposure",            str(cfg.get("exposure_ms","–")) + " ms"],
        ["Lighting",            cfg.get("lighting","–")],
        ["mm per pixel",        str(cfg.get("mm_per_pixel","–"))],
        ["Mean latency",        str(cfg.get("mean_latency_s","–")) + " s"],
        ["Defect recall",       str(cfg.get("defect_recall","–"))],
    ]
    story.append(tbl(design_data, col_widths=[6*cm, 10*cm]))
    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph("Assumptions", style_h2))
    for a in cfg.get("assumptions",[]):
        story.append(Paragraph(f"• {a}", style_body))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("Failure Cases", style_h2))
    for f_case in cfg.get("failure_cases",[]):
        story.append(Paragraph(f"• {f_case}", style_body))

# Build PDF
doc.build(story)
print("Report saved → outputs/Assignment1_Report.pdf")
