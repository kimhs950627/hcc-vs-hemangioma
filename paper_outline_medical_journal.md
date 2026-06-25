# Paper Outline — Medical Journal Submission

> **Status**: Draft outline (pre-results)  
> **Target journal tier**: PubMed-indexed, SCIE Q1–Q2 (e.g., *Ultrasonics*, *Diagnostics*, *Frontiers in Oncology*, *JMIR Medical Informatics*)

---

## Proposed Title

**"A Hybrid Vision Transformer Trained with VICReg Self-Supervised Learning for Operator-Independent HCC Scoring from B-mode Ultrasound: A Single-Center Retrospective Study"**

*Alternative (shorter):*  
"Deep Learning-Based HCC Score from B-mode Ultrasound as an Operator-Independent Radiologic Marker: A Single-Center Retrospective Study"

---

## Structured Abstract

| Section | Content |
|---|---|
| **Background** | B-mode ultrasound (US) is the recommended first-line surveillance tool for hepatocellular carcinoma (HCC), yet operator dependency and limited sensitivity for early-stage lesions (~45–63%) remain major obstacles. In primary care settings, tumor markers such as AFP and PIVKA-II are often inaccessible due to reimbursement and logistical barriers. An image-only, operator-independent radiologic score is therefore clinically desirable. |
| **Methods** | Single-center retrospective study. B-mode US images of HCC and hepatic hemangioma were split into Train/Val/Test (1,858/530/268). A Hybrid Vision Transformer (CNN stem + Transformer encoder) was self-supervised with VICReg, then fine-tuned with a linear classifier. HCC Score = P(HCC) ∈ [0,1] derived from softmax output. Optimal cutoff determined on Val set (Youden's J); Test set was blind. Performance reported as AUROC (DeLong 95% CI), Sensitivity, Specificity, PPV, NPV, and Decision Curve Analysis (DCA). |
| **Results** | *(To be filled after training)* Test AUROC [XX] (95% CI [XX–XX]). At Val-derived cutoff: Sensitivity XX%, Specificity XX%, PPV XX%, NPV XX%. DCA demonstrated net benefit over treat-all/treat-none in the [XX–XX]% threshold probability range. |
| **Conclusions** | A VICReg-pretrained Hybrid ViT produces a continuous, operator-independent HCC Score from B-mode US alone, supporting its potential utility as a primary-care radiologic marker. Future studies should incorporate serology (AFP, PIVKA-II), multi-center prospective cohorts, and radiologist-level comparisons. |

**Keywords**: hepatocellular carcinoma; ultrasound; deep learning; self-supervised learning; VICReg; vision transformer; radiologic marker; primary care

---

## 1. Introduction

### 1.1 Clinical Burden of HCC

- Hepatocellular carcinoma (HCC) accounts for 85–90% of primary liver cancers and ranks third in global cancer mortality.
- HBV and HCV infection are the dominant etiologies; incidence is disproportionately high in East Asia including South Korea.
- Early detection is the single most impactful modifiable factor: 5-year survival rate exceeds 70% within Milan criteria (resection or transplant), versus <10% at advanced stage.
- Current international guidelines (AASLD, EASL, APASL) recommend 6-month interval B-mode ultrasound surveillance with or without AFP for high-risk populations (cirrhosis, chronic hepatitis B/C).

### 1.2 Screening Limitations and the Primary Care Gap

- Ultrasound sensitivity for early-stage HCC (≤2 cm): approximately 45% in isolation, rising to 63% when combined with AFP (Tzartzeva et al., *Gastroenterology* 2018).
- Operator dependency: image quality and interpretation vary substantially between examiners, limiting reproducibility especially in primary care clinics and smaller hospitals without on-site hepatologists or radiologists.
- **Primary care reality**: In South Korean primary care and comparable healthcare systems, PIVKA-II and AFP testing is often constrained by insurance reimbursement criteria, referral requirements, and patient access. Consequently, tumor-marker-independent screening tools are of high practical value.
- AFP-negative HCC constitutes approximately two-thirds of all HCC cases, rendering AFP-based screening inherently incomplete (Tsuchiya et al., *WJG* 2015).

### 1.3 Artificial Intelligence in Liver Ultrasound: Opportunity and Gap

- Deep learning (DL) models have demonstrated AUC 0.83–0.94 for focal liver lesion (FLL) classification on B-mode US (Yang et al., *eBioMedicine* 2020; Zhang et al., *Front Oncol* 2022).
- However, existing studies share critical methodological limitations:
  - Binary (HCC/non-HCC) outputs that preclude use as a continuous radiologic marker.
  - Threshold determination from test data — a form of data leakage that inflates reported performance.
  - Absence of Decision Curve Analysis (DCA) or other clinical utility assessments.
  - Use of plain Vision Transformers (ViT) with limited adaptation to the small-dataset reality of medical imaging.
- Self-supervised learning (SSL) has emerged as a powerful pretraining strategy when labeled medical data is scarce. VICReg (Bardes et al., *NeurIPS 2022*) offers stable training without a momentum teacher, with an explicit covariance regularization term enforcing feature decorrelation — a desirable property for texturally similar lesion images.

### 1.4 Study Objectives

1. Develop a **HCC Score** (continuous 0–1 probability) from B-mode US images using a VICReg-pretrained Hybrid Vision Transformer.
2. Establish a rigorous cutoff determination protocol: **Val-set-only Youden's J**, with the Test set preserved as a blind evaluation.
3. Evaluate the model as a radiologic marker via AUROC, per-cutoff diagnostics, and DCA.
4. Situate findings in the primary care context and delineate methodological contributions and study limitations.

---

## 2. Materials and Methods

### 2.1 Study Design

- **Design**: Single-center, retrospective, observational study.
- **Reporting**: TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) guideline.
- **Ethics**: IRB approval obtained (approval number: *to be inserted*).
- **Setting**: *[Institution name, city, country]* — *[period of data collection]*.

### 2.2 Participants and Image Acquisition

**Inclusion criteria**:
- Histologically or imaging-confirmed HCC meeting AASLD/EASL non-invasive criteria, OR
- Hemangioma confirmed by contrast-enhanced CT/MRI or follow-up stability.
- Available B-mode ultrasound image in the institutional PACS archive.

**Exclusion criteria**:
- Poor image quality (excessive shadowing, patient motion artifact).
- Indeterminate or equivocal diagnosis.
- *(Institution-specific criteria to be inserted)*.

**Dataset composition**:

| Split | Total | HCC | Hemangioma |
|---|---|---|---|
| Train | 1,858 | 972 (52.3%) | 886 (47.7%) |
| Val | 530 | 277 (52.3%) | 253 (47.7%) |
| Test | 268 | 140 (52.2%) | 128 (47.8%) |
| **Total** | **2,656** | **1,389** | **1,267** |

- Image specification: 384 × 384 px, grayscale, float32 [0, 255], normalize=False.
- **Split level**: *(Confirm patient-level vs. image-level split — document to avoid data leakage disclosure)*.

### 2.3 Model Architecture: VICReg + Hybrid ViT

#### 2.3.1 Hybrid Vision Transformer (Encoder)

A Hybrid ViT replaces the standard linear patch embedding of plain ViT with a convolutional stem:

- **CNN Stem**: Conv2D (kernel 7×7, stride 4, channels 96) → LayerNorm → produces feature maps that serve as patch tokens.
- **Transformer Encoder**: ViT-Small configuration — depth=12, heads=6, embedding dimension=384.
- **Output**: [CLS] token representation, shape (B, 384).

The CNN stem provides local translation-equivariant feature extraction before global self-attention, which is critical for small labeled datasets where plain ViT's attention maps tend to under-specialize.

#### 2.3.2 VICReg Self-Supervised Pre-training

VICReg (Variance–Invariance–Covariance Regularization) learns representations by simultaneously optimizing three terms:

- **Invariance loss**: Mean squared error between two views of the same image — encourages view-consistent representations.
- **Variance loss**: Hinge on per-dimension standard deviation (target: std ≥ 1) — prevents feature collapse.
- **Covariance loss**: Penalizes off-diagonal entries of the feature covariance matrix — enforces decorrelated dimensions.

```
L_VICReg = λ · L_invariance + μ · L_variance + ν · L_covariance
Default: λ=25, μ=25, ν=1  (Bardes et al., NeurIPS 2022)
```

A symmetric twin network (two identical encoders sharing weights) processes two random augmented views (v1, v2) of each US image. No momentum teacher or stop-gradient is required, simplifying training.

**Augmentation strategy (SSL pre-training)**:
- Random crop and resize to 384 × 384
- Random horizontal and vertical flip
- Gaussian blur (σ ∈ [0.1, 2.0])
- Brightness and contrast jitter (restricted to achromatic channels for grayscale US; no hue/saturation)

**Projector MLP**: 384 → 2048 → 2048 → 2048 (BatchNorm + ReLU; final layer has no activation). Discarded at fine-tuning.

#### 2.3.3 Supervised Fine-tuning (PureClassifier)

After pre-training, the Hybrid ViT encoder is attached to a linear classification head:

- **Head**: Dense(2) → Softmax
- **Output**: `{"logits": (B, 2), "probabilities": (B, 2)}`
- **Fine-tuning strategy**: Encoder partially frozen (last N transformer blocks unfrozen) — exact block count determined empirically.
- **Loss**: Categorical cross-entropy.
- **Optimizer, LR, schedule**: *(to be inserted after training)*.

### 2.4 HCC Score Definition

> **HCC Score = probabilities[:, 1] = P(HCC) ∈ [0, 1]**

- This continuous output is treated as a **radiologic marker** analogous to a serum biomarker — not merely a binary classifier.
- Score ↑ indicates increasing probability of HCC; Score ↓ favors hemangioma.
- The score itself carries clinical information independent of any binary threshold.

### 2.5 Cutoff Determination (No Data Leakage)

Optimal cutoff is determined exclusively from the **Validation set**:

| Strategy | Definition | Role |
|---|---|---|
| **Youden's J** (primary) | argmax (Sensitivity + Specificity − 1) on Val ROC curve | Primary reported cutoff |
| Sensitivity-first (secondary) | Lowest cutoff satisfying Sensitivity ≥ 0.90 on Val ROC | Sensitivity analysis |

**Rationale**: Determining thresholds from test data constitutes data leakage and violates the statistical independence required for unbiased generalization estimation. In a clinical deployment scenario, a cutoff must be fixed *before* seeing new patients; the Val-set-only approach replicates this prospective constraint.

### 2.6 Statistical Analysis

- **AUROC**: DeLong method with 95% confidence intervals (DeLong et al., *Biometrics* 1988).
- **At cutoff**: Sensitivity, Specificity, PPV, NPV, F1-score, confusion matrix (TP/FP/TN/FN).
- **Decision Curve Analysis (DCA)**: Net benefit across threshold probability range 0.05–0.95 against treat-all and treat-none strategies (Vickers & Elkin, *Med Decis Making* 2006).
- **Bootstrap resampling** (n = 1,000): 95% CI for all metrics on Test set.
- **Score distribution**: Per-class Gaussian KDE plot for Train, Val, and Test separately.
- Software: Python 3.11, scikit-learn, scipy, matplotlib.

---

## 3. Results *(To be completed after training)*

### 3.1 Study Population and Image Characteristics (Table 1)

| Characteristic | Train | Val | Test |
|---|---|---|---|
| Total images, n | 1,858 | 530 | 268 |
| HCC, n (%) | 972 (52.3%) | 277 (52.3%) | 140 (52.2%) |
| Hemangioma, n (%) | 886 (47.7%) | 253 (47.7%) | 128 (47.8%) |
| Age, mean ± SD, years | — | — | — |
| Male sex, n (%) | — | — | — |
| Underlying cirrhosis, n (%) | — | — | — |
| Lesion size, cm (median [IQR]) | — | — | — |

### 3.2 VICReg Pre-training Convergence

- Invariance, Variance, Covariance loss curves across epochs.
- Final pre-training loss values.
- Linear probing AUROC on Val at end of pre-training (before supervised fine-tuning) — demonstrating representation quality.

### 3.3 HCC Score Distribution (Figure 1)

- Three-row subplot: Train / Val / Test.
- Per row: Gaussian KDE for Hemangioma (blue) vs. HCC (red).
- Vertical dashed line at Val-derived cutoff.
- Degree of class separation assessed qualitatively and by AUROC.

### 3.4 ROC Curves and AUROC (Figure 2)

- Overlay of Train, Val, and Test ROC curves.
- Val cutoff operating point marked (sensitivity, 1−specificity).
- DeLong 95% CI for Test AUROC.

### 3.5 Performance at Val-Derived Cutoff (Table 2)

| Metric | Val (cutoff derivation) | Test (blind evaluation) |
|---|---|---|
| AUROC (95% CI) | — | — |
| Cutoff (Youden's J) | — | Same as Val |
| Sensitivity (%) | — | — |
| Specificity (%) | — | — |
| PPV (%) | — | — |
| NPV (%) | — | — |
| F1-score | — | — |
| TP / FP / TN / FN | — | — |

### 3.6 Decision Curve Analysis (Figure 3)

- X-axis: Threshold probability (0.05–0.95).
- Y-axis: Net benefit.
- Three lines: HCC Score model / Treat-all / Treat-none.
- Annotation of threshold probability range where model net benefit > treat-all.

---

## 4. Discussion

### 4.1 Principal Findings

This study presents the first application of VICReg self-supervised learning with a Hybrid Vision Transformer for continuous HCC scoring from B-mode ultrasound. The HCC Score achieved Test AUROC [XX] (95% CI [XX–XX]), demonstrating that an image-only deep learning model can meaningfully discriminate HCC from hemangioma without relying on tumor markers, contrast agents, or Doppler data.  
At the Val-derived cutoff, Sensitivity and Specificity were [XX]% and [XX]%, respectively — a performance profile that is clinically interpretable in the same way as AFP or PIVKA-II thresholds in conventional practice.

### 4.2 Clinical Relevance in the Primary Care Context

The primary care physician who performs abdominal ultrasound routinely does not have on-site access to radiologist interpretation, and tumor marker testing is frequently constrained by reimbursement policies and referral pathways. The HCC Score requires only the B-mode US image — a resource universally available in primary care clinics — and produces a continuous output rather than a binary flag. This operator-independent, marker-independent output can serve as an adjunct screening layer that stratifies patients who warrant expedited hepatology referral, without replacing existing biomarker-based protocols where those are available.

### 4.3 Methodological Contribution: Val-Only Cutoff Strategy

A substantial proportion of published DL studies in liver US determine diagnostic thresholds on test data — either explicitly or implicitly via repeated threshold testing. This practice inflates reported sensitivity/specificity and renders the quoted cutoff clinically meaningless, because it would not generalize to an independent prospective cohort. By confining cutoff derivation to the validation set and preserving the test set as a single-use blind evaluation, this study replicates the statistical architecture of a prospective clinical trial. Future studies reporting DL-based radiologic markers should adopt this or an equivalent methodology.

### 4.4 VICReg and Hybrid ViT: Architectural Rationale

The choice of VICReg over contrastive methods (SimCLR) or momentum-based SSL (DINO, MoCo) reflects practical constraints of the medical imaging domain: small labeled sets, limited compute (single consumer GPU), and the need for training stability. VICReg's explicit variance and covariance loss terms act as regularizers that are particularly valuable when the feature space dimensionality (2048) exceeds the effective batch size — a common scenario in medical imaging. The Hybrid ViT resolves the data-hungry nature of plain ViT by introducing CNN-derived inductive biases (locality, translation equivariance) that reduce dependence on large-scale pretraining corpora, while the Transformer encoder retains global context modeling for lesion-level reasoning.

### 4.5 Clinical Utility: Decision Curve Analysis

DCA net benefit in the [XX]–[XX]% threshold probability range indicates that applying the HCC Score model in patients whose pre-test probability of HCC falls within that range would yield more true-positive referrals per false-positive referral compared to both the treat-all and treat-none strategies. This quantification of net clinical benefit is absent from the majority of prior DL-based liver US studies and constitutes a necessary bridge between model performance metrics (AUROC) and actual clinical decision-making utility.

### 4.6 Comparison with Prior Work

| Study | Modality | Classes | AUROC | Cutoff Method | DCA | SSL |
|---|---|---|---|---|---|---|
| Yang 2020 (*eBioMedicine*) | B-mode US | FLL (multi-class) | 0.83–0.94 | Test-derived | No | No |
| Zhang 2022 (*Front Oncol*) | CEUS | AFP-neg HCC vs FNH | 0.937 | Unclear | No | No |
| **This study** | B-mode US | HCC vs Hemangioma | [XX] | **Val-only (Youden)** | **Yes** | **VICReg** |

Key differentiators: (1) continuous score as radiologic marker, (2) methodologically rigorous val-only cutoff, (3) DCA for clinical utility, (4) VICReg SSL suited to small labeled datasets, (5) explicit primary care applicability framing.

### 4.7 Limitations

1. **Single-center retrospective design**: External generalizability to different ultrasound equipment, operator practices, and patient populations is unvalidated.
2. **Absence of radiologist comparison**: No human baseline performance data.
3. **No tumor marker data**: AFP-negative HCC subgroup analysis (approximately two-thirds of HCC cases) is not possible within the current dataset.
4. **Lesion size metadata**: Without size stratification, performance in early-stage HCC (≤2 cm) cannot be separately assessed.
5. **Binary control group**: The comparator is hemangioma only. Real-world differential includes FNH, regenerative nodules, and metastases.
6. **Image-level split risk**: If the dataset was not split at the patient level, images from the same patient may appear across splits, introducing optimistic bias.

---

## 5. Conclusion

A Hybrid Vision Transformer pretrained with VICReg self-supervised learning produces a continuous HCC Score from B-mode ultrasound alone, achieving Test AUROC [XX] and clinically meaningful sensitivity/specificity at a methodologically rigorous, Val-set-derived cutoff. Decision Curve Analysis demonstrates net clinical benefit over treat-all and treat-none strategies within the [XX]–[XX]% threshold probability range. The model requires no tumor markers, no contrast agents, and no Doppler capability, making it directly applicable in primary care and resource-limited settings. Limitations of single-center retrospective design and absent tumor marker comparisons motivate three priority future directions: (1) paired AFP/PIVKA-II validation in an AFP-negative subgroup, (2) multi-center prospective cohort study, and (3) head-to-head comparison with radiologist-level performance.

---

## 6. Key References

| # | Citation | Section |
|---|---|---|
| 1 | Tzartzeva K et al. *Gastroenterology* 2018 — US+AFP sensitivity meta-analysis | Intro 1.2, Discussion 4.5 |
| 2 | EASL Clinical Practice Guidelines: HCC. *J Hepatol* 2018 | Methods 2.1, Intro 1.1 |
| 3 | AASLD HCC Guidance 2018 | Intro 1.1, 1.2 |
| 4 | Bardes A et al. *NeurIPS 2022* — VICReg | Methods 2.3.2, Discussion 4.4 |
| 5 | Yang Q et al. *eBioMedicine* 2020 — multicenter FLL DL | Intro 1.3, Discussion 4.6 |
| 6 | Zhang W-B et al. *Front Oncol* 2022 — AFP-neg HCC/FNH DL | Intro 1.3, Discussion 4.6 |
| 7 | Vickers AJ & Elkin EB. *Med Decis Making* 2006 — DCA methodology | Methods 2.6, Discussion 4.5 |
| 8 | DeLong ER et al. *Biometrics* 1988 — AUROC 95% CI | Methods 2.6 |
| 9 | Collins GS et al. *BMJ* 2015 — TRIPOD guideline | Methods 2.1 |
| 10 | Tsuchiya N et al. *WJG* 2015 — AFP-negative HCC prevalence | Intro 1.2, Limitation 4.7 |
| 11 | Dosovitskiy A et al. *ICLR 2021* — ViT | Methods 2.3.1 |
| 12 | Kolesnikov A et al. *ECCV 2020* — Hybrid ViT | Methods 2.3.1 |

---

## Appendix: Planned Figures and Tables

| Item | Description | Status |
|---|---|---|
| **Figure 1** | HCC Score KDE distribution (Train/Val/Test, 3-row, with cutoff line) | Pending training |
| **Figure 2** | ROC curves overlay (Train/Val/Test) + Val cutoff operating point | Pending training |
| **Figure 3** | Decision Curve Analysis (Model vs Treat-all vs Treat-none) | Pending training |
| **Table 1** | Study population characteristics (age, sex, cirrhosis, lesion size) | Pending metadata |
| **Table 2** | Diagnostic performance at Val-derived cutoff (Val vs Test) | Pending training |
| **Supp. Fig. 1** | VICReg pre-training loss curves (Invariance/Variance/Covariance) | Pending training |
| **Supp. Table 1** | Sensitivity-first cutoff analysis (Sensitivity ≥ 0.90) | Pending training |

---

*Last updated: 2026-06-25*  
*Author: Kim Hyun-Soo, M.D. — Department of Family Medicine, Jeonju, Republic of Korea*
