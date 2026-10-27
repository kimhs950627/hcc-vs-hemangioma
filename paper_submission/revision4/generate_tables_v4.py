"""
Generate revised Tables and Figure Captions in Word format (docx) for KJFP revision.
File: tables_and_figures_revised.docx
"""

import os
import pathlib
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

OUT_DIR = str(pathlib.Path(__file__).resolve().parent)
os.makedirs(OUT_DIR, exist_ok=True)

doc = docx.Document()
for s in doc.sections:
    s.top_margin = Inches(1.0)
    s.bottom_margin = Inches(1.0)
    s.left_margin = Inches(1.0)
    s.right_margin = Inches(1.0)

BLUE = RGBColor(0, 0, 220)

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def add_table_header(table, headers):
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.bold = True
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(9.5)
        set_cell_background(hdr_cells[i], "F0F0F0")

def add_caption(doc, text, is_revised=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(text)
    run.font.bold = True
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(11)
    if is_revised:
        run.font.color.rgb = BLUE

def add_footnote(doc, text, is_revised=False):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(10)
    run = p.add_run(text)
    run.font.name = "Malgun Gothic"
    run.font.size = Pt(8.5)
    run.font.italic = True
    if is_revised:
        run.font.color.rgb = BLUE

# --- Table 1 ---
add_caption(doc, "Table 1. Patient and Lesion Demographics of Full Cohort (SMC-LUD, n=1,021)", is_revised=True)
t1_data = [
    ["항목 (Characteristics)", "HCC (환자 n=600)", "Hemangioma (환자 n=421)"],
    ["나이, 평균 ± 표준편차 (범위), 세", "66.65 ± 11.02 (20–90)", "55.50 ± 12.76 (30–90)"],
    ["남성, n (%)", "491 (81.8%)", "174 (41.3%)"],
    ["여성, n (%)", "109 (18.2%)", "247 (58.7%)"],
    ["키, 평균 ± 표준편차, cm", "165.74 ± 16.39", "163.72 ± 8.45"],
    ["몸무게, 평균 ± 표준편차 (범위), kg", "67.44 ± 11.24 (42–104)", "63.85 ± 12.95 (34–118)"],
    ["병변 크기 (최대 직경), 중앙값 / Q1 / Q3, cm", "2.90 / 2.10 / 4.50", "—"],
    ["UICC 병기 (2000), n (%)", "", ""],
    ["  pT1", "58 (9.7%)", "—"],
    ["  pT2", "120 (20.0%)", "—"],
    ["  pT3", "353 (58.8%)", "—"],
    ["  pT4", "69 (11.5%)", "—"],
    ["AJCC 병기 (2010), n (%)", "", ""],
    ["  pT1", "164 (27.3%)", "—"],
    ["  pT2", "393 (65.5%)", "—"],
    ["  pT3", "22 (3.7%)", "—"],
    ["  pT4", "21 (3.5%)", "—"],
]

t1 = doc.add_table(rows=len(t1_data), cols=3)
t1.style = 'Table Grid'
add_table_header(t1, t1_data[0])
for r_idx, row in enumerate(t1_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t1.rows[r_idx].cells[c_idx]
        cell.text = val
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(9)
add_footnote(doc, "Data source: Tak et al. Sci Data (2026)[13]. Values represent full SMC-LUD cohort (n=1,021 patients). Note that tumor diameter and pathological stages were only documented for HCC cases in the original public repository.", is_revised=True)

# --- Table 1B ---
add_caption(doc, "Table 1B. Dataset Allocation for Clean Subset (744 Patients, 2,656 Images)", is_revised=False)
t1b_data = [
    ["Diagnosis", "No. of Patients", "No. of Images", "Train Set", "Validation Set", "Test Set"],
    ["Hemangioma", "323", "1,267", "886", "253", "128"],
    ["HCC", "421", "1,389", "972", "277", "140"],
    ["Total (Clean)", "744", "2,656", "1,858", "530", "268"]
]
t1b = doc.add_table(rows=len(t1b_data), cols=6)
t1b.style = 'Table Grid'
add_table_header(t1b, t1b_data[0])
for r_idx, row in enumerate(t1b_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t1b.rows[r_idx].cells[c_idx]
        cell.text = val
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(9)
            if r_idx == len(t1b_data)-1:
                run.font.bold = True
add_footnote(doc, "Values follow the patient-level split published by Tak et al. (2026). Images with calipers or measuring overlays were excluded to prevent shortcut feature learning.", is_revised=True)

# --- Table 2 ---
add_caption(doc, "Table 2. Ablation: Backbone × Training Mode (Validation Set, Confidence Score)", is_revised=True)
t2_data = [
    ["#", "Backbone", "Training Mode", "Cutoff", "AUROC", "Sensitivity (%)", "Specificity (%)", "F1 Score"],
    ["1", "ResNet50V2", "CE Only", "0.0002", "1.000", "100.00", "100.00", "1.0000"],
    ["2", "EfficientNetV2B0", "CE Only", "0.0016", "0.9999", "100.00", "99.60", "0.9982"],
    ["3", "ResNet50V2", "CE + SupCon", "0.0002", "1.000", "99.64", "100.00", "0.9982"],
    ["4", "EfficientNetV2B0*", "CE + SupCon", "0.0011", "1.000", "100.00", "100.00", "1.0000"],
    ["5", "ResNet50V2", "NNCLR (SSL)", "0.0003", "0.9994", "99.64", "99.21", "0.9946"],
    ["6", "EfficientNetV2B0", "NNCLR (SSL)", "0.0038", "0.9986", "98.19", "98.42", "0.9837"],
]
t2 = doc.add_table(rows=len(t2_data), cols=8)
t2.style = 'Table Grid'
add_table_header(t2, t2_data[0])
for r_idx, row in enumerate(t2_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t2.rows[r_idx].cells[c_idx]
        cell.text = val
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(8.5)
            if r_idx == 4: # primary model
                run.font.bold = True
add_footnote(doc, "*Primary model (#4). Cutoffs were determined on the validation set using Youden's J statistic. All values are recomputed from raw prediction logs. CE, Cross-Entropy; SupCon, Supervised Contrastive Learning; NNCLR, Nearest-Neighbor Contrastive Learning (2-stage: SSL pre-training followed by CE+SupCon fine-tuning). Cutoffs in this ablation table are validation-derived Youden points used only to compare models; the operating point of the primary model is reported in Table 3 (confidence score >= 0.50).", is_revised=True)

# --- Table 3 (primary operating point, test set only) ---
add_caption(doc, "Table 3. Primary Model Classification Performance at the Primary Operating Point (Confidence Score >= 0.50), Test Set", is_revised=True)
t3_data = [
    ["Metric", "Test Set (n=268)"],
    ["AUROC", "1.000"],
    ["Accuracy (%) (95% CI)", "97.76 (95.18-98.96)"],
    ["Sensitivity (%) (95% CI)", "95.71 (90.98-98.00)"],
    ["Specificity (%) (95% CI)", "100.00 (97.10-100.00)"],
    ["Positive Predictive Value (%)", "100.00"],
    ["Negative Predictive Value (%)", "95.52"],
    ["F1 Score", "0.9781"],
    ["Confusion Matrix (TP / FP / FN / TN)", "134 / 0 / 6 / 128"]
]
t3 = doc.add_table(rows=len(t3_data), cols=2)
t3.style = 'Table Grid'
add_table_header(t3, t3_data[0])
for r_idx, row in enumerate(t3_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t3.rows[r_idx].cells[c_idx]
        cell.text = val
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(9)
add_footnote(doc, "Performance is reported for the test set only; validation-set performance is not reported because thresholds were derived there (in-sample). The primary operating point was prespecified at a confidence score of 0.50 rather than a validation-derived Youden point, because AUROC reached the ceiling (1.000) in the derivation cohort. Test set: n=268 (HCC 140, hemangioma 128). Misclassifications at this operating point were false negatives only. The 0.50 cutoff lies inside its validation equivalence interval (0.4498, 0.5400]; the six test HCC cases missed at this operating point have predicted probabilities of 0.029-0.404. 95% confidence intervals are Wilson score intervals estimated at the image level.", is_revised=True)

# --- Table 3B (similarity operating points, test set only) ---
add_caption(doc, "Table 3B. Exploratory Operating Points of the Similarity Metrics, Test Set", is_revised=True)
t3b_data = [
    ["Output", "Cutoff (derived on Validation)", "Validation equivalence interval", "Test Sens / Spec (%)", "Test Acc (%)", "Test TP/FP/FN/TN"],
    ["HCC cosine score (mean)", "-0.004843", "(-0.0498, -0.0048]", "100.00 / 99.22", "99.63", "140 / 1 / 0 / 127"],
    ["Delta score (mean)", "-0.958471", "(-1.0001, -0.9585]", "100.00 / 99.22", "99.63", "140 / 1 / 0 / 127"]
]
t3b = doc.add_table(rows=len(t3b_data), cols=6)
t3b.style = 'Table Grid'
add_table_header(t3b, t3b_data[0])
for r_idx, row in enumerate(t3b_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t3b.rows[r_idx].cells[c_idx]
        cell.text = val
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT if c_idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
        for run in p.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(8.5)
add_footnote(doc, "Cutoffs and their equivalence intervals were both derived on the validation set; all performance and confusion matrices are reported for the test set only. Equivalence interval = the range of validation thresholds that reproduces the same validation confusion matrix, so it always contains the reported cutoff. Because AUROC reached 1.000 in the derivation cohort the optimum is not unique, and for both similarity scores the cutoff coincides with the upper bound of the interval, which is set by the single lowest-scoring HCC case in the validation set. Both similarity outputs produced the same single false positive, which is classified correctly at the primary 0.50 operating point. Discordant classifications versus the primary operating point: 6 versus 1 (McNemar exact p = 0.125); neither superiority nor non-inferiority is claimed. Thresholds require re-derivation in an external cohort.", is_revised=True)

# --- Table 4 (merged: operating points + cross-model AUROC, test set only) ---
add_caption(doc, "Table 4. Dual-Output Metrics in the Primary Model and Test-Set AUROC across All Evaluated Models", is_revised=True)
add_footnote(doc, "A. Operating-point performance of the primary model (EfficientNetV2B0 + CE+SupCon), test set (n = 268)", is_revised=True)

t4_data = [
    ["Output", "Metric", "Cutoff", "AUROC", "Sensitivity (%)", "Specificity (%)", "Errors (FP / FN)"],
    ["ROC-A", "Confidence score", "0.50 (prespecified)", "1.000", "95.71", "100.00", "0 / 6"],
    ["ROC-B", "HCC cosine score, mean prototype", "-0.004843", "1.000", "100.00", "99.22", "1 / 0"],
    ["ROC-B", "HCC cosine score, k-means (k=4)", "0.0288", "1.000", "100.00", "99.22", "1 / 0"],
    ["ROC-C", "Delta score, mean prototype", "-0.958471", "1.000", "100.00", "99.22", "1 / 0"],
    ["ROC-C", "Delta score, k-means (k=4)", "-0.9265", "1.000", "100.00", "99.22", "1 / 0"]
]
t4 = doc.add_table(rows=len(t4_data), cols=7)
t4.style = 'Table Grid'
add_table_header(t4, t4_data[0])
for r_idx, row in enumerate(t4_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t4.rows[r_idx].cells[c_idx]
        cell.text = val
        p_ = cell.paragraphs[0]
        p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p_.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(8)

add_footnote(doc, "B. Test-set AUROC across all evaluated models (n = 268)", is_revised=True)
t4b_data = [
    ["Backbone", "Training mode", "Confidence (A)", "Cosine-B (mean)", "Cosine-B (k-means)", "Delta-C (mean)", "Delta-C (k-means)"],
    ["EfficientNetV2B0", "CE only", "0.9999", "0.9999", "1.0000", "0.9999", "0.9999"],
    ["EfficientNetV2B0*", "CE + SupCon", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["EfficientNetV2B0", "NNCLR (SSL)", "1.0000", "0.8930", "0.9763", "0.9996", "0.9999"],
    ["ResNet50V2", "CE only", "1.0000", "1.0000", "1.0000", "1.0000", "1.0000"],
    ["ResNet50V2", "CE + SupCon", "0.9999", "0.9999", "0.9999", "0.9999", "0.9999"],
    ["ResNet50V2", "NNCLR (SSL)", "0.9994", "0.0684", "0.1235", "0.9891", "0.9984"]
]
t4b = doc.add_table(rows=len(t4b_data), cols=7)
t4b.style = 'Table Grid'
add_table_header(t4b, t4b_data[0])
for r_idx, row in enumerate(t4b_data[1:], start=1):
    for c_idx, val in enumerate(row):
        cell = t4b.rows[r_idx].cells[c_idx]
        cell.text = val
        p_ = cell.paragraphs[0]
        p_.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in p_.runs:
            run.font.name = "Malgun Gothic"
            run.font.size = Pt(8)
            if r_idx == 2:
                run.font.bold = True
add_footnote(doc, "All prototypes were derived strictly from the training set. Test-set results only. ROC-A uses the prespecified conventional operating point (confidence score >= 0.50); ROC-B and ROC-C use validation Youden points and are exploratory. Because every AUROC sits at 1.000, DeLong testing is uninformative and no non-inferiority claim is made. At the similarity operating points a single hemangioma image (index 576) was misclassified identically by all four similarity definitions, whereas at the 0.50 operating point the six errors were false negatives; McNemar exact p = 0.125. *Primary model. In ResNet50V2 + NNCLR the single-prototype cosine AUROC collapsed to 0.0684 (label-space inversion from unguided self-supervised pre-training).", is_revised=True)

# --- Figure Captions ---
p_fig = doc.add_paragraph()
p_fig.paragraph_format.space_before = Pt(20)
run_fig = p_fig.add_run("그림 설명 (Figure Captions)")
run_fig.font.bold = True
run_fig.font.name = "Malgun Gothic"
run_fig.font.size = Pt(13)

from captions_v4 import CAPTIONS as captions


for title, desc, is_rev in captions:
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(2)
    r1 = p.add_run(title + "\n")
    r1.font.bold = True
    r1.font.name = "Malgun Gothic"
    r1.font.size = Pt(10)
    if is_rev:
        r1.font.color.rgb = BLUE
        
    r2 = p.add_run(desc)
    r2.font.name = "Malgun Gothic"
    r2.font.size = Pt(9.5)
    if is_rev:
        r2.font.color.rgb = BLUE

doc_path = os.path.join(OUT_DIR, "tables_v4.docx")
doc.save(doc_path)
print(f"Saved revised tables and figure captions to: {doc_path}")
