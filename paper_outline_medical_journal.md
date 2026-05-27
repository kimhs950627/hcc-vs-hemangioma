# Proposed Medical-Journal Manuscript Outline for Ultrasound AI in HCC vs Hemangioma

## Working title

**Lightweight Self-Supervised Ultrasound Encoder for Differentiating Hepatocellular Carcinoma and Hemangioma on B-Mode Liver Ultrasound: Emphasis on Patch-Level Representation Quality and Headwise Attention Specialization**

## Target style

This manuscript should follow a **medical imaging / clinical AI paper format**, not a machine-learning conference style. Recent abdominal ultrasound and focal liver lesion papers in medical journals typically use the structure of **Introduction, Materials and Methods, Results, Discussion, Conclusion**, with explicit clinical motivation, patient/image cohort description, reference standard, statistical analysis, and limitations.[web:1042][web:1044][web:1041]

## Why this study matters

Prior medical-journal literature has shown that deep learning on abdominal ultrasound can detect, localize, or classify focal liver lesions and may perform comparably to experts in selected settings.[web:1042][web:1044] More recent studies also suggest that ultrasound-based machine learning can improve diagnosis of small HCC and that public liver ultrasound datasets are now emerging, which makes reproducible development more feasible.[web:1045][web:1030] However, many published studies focus on fully supervised end tasks, CEUS, object detection, or image-level classification rather than on building a reusable **stage-1 ultrasound representation** that preserves patch-level information for downstream tasks and prototype reasoning.[web:1041][web:1042][web:1051]

## Related medical-journal literature and relevance

| Paper | Journal / Year | Main task | Relevance to this project | Gap relative to this project |
|---|---|---|---|---|
| Deep Learning for the Detection, Localization, and Characterization of Focal Liver Lesions on Abdominal US Images | Radiology: AI, 2022 [web:1042] | Detection, localization, benign-vs-malignant characterization on B-mode US | Shows strong clinical framing, expert comparison, and abdominal US relevance | Supervised detection pipeline; not focused on reusable SSL encoder or HCC-vs-hemangioma prototype learning |
| Deep learning for differentiation of benign and malignant solid liver lesions on ultrasonography | Abdominal Radiology, 2021 [web:1044] | Benign-vs-malignant differentiation | Clinically close task and likely reviewer anchor paper | Binary malignant-vs-benign framing rather than specific HCC-vs-hemangioma SSL representation design |
| Malignancy diagnosis of liver lesion in contrast enhanced ultrasound using an end-to-end method based on deep learning | BMC Medical Imaging, 2024 [web:1041] | CEUS malignancy diagnosis | Shows medical-journal writing style and current liver-lesion AI interest | CEUS video setting differs from grayscale B-mode and from SSL pretraining emphasis |
| An Automated Method for Classifying Liver Lesions in Contrast-Enhanced Ultrasound Imaging Based on Deep Learning Algorithms | Diagnostics, 2023 [web:1043] | CEUS lesion classification | Useful comparator for liver lesion classification framing | Uses CEUS and modular supervised pipeline; less relevant to lightweight grayscale SSL encoder |
| Development and validation of an ultrasound-based model for small HCC | eClinicalMedicine, 2025 [web:1045] | Grayscale US model for small HCC | Strong clinical precedent for grayscale ultrasound-based HCC diagnosis | Focuses on small-HCC diagnosis, not HCC-vs-hemangioma representation learning or prototype induction |
| SMC-LUD: Large-Scale B-Mode Liver Ultrasound Dataset | Scientific Data, 2026 [web:1030] | Public dataset release with HCC and hemangioma classes | Directly relevant because it demonstrates dataset feasibility and class pair relevance | Dataset paper rather than method paper; does not solve stable SSL representation learning |
| Interpretable Machine Learning for Characterization of Focal Liver Lesions on CEUS | Scientific Reports, 2022 [web:1051] | Interpretable CEUS-based lesion characterization | Supports the need for interpretability and clinically meaningful features | CEUS/radiomics pipeline, not grayscale B-mode SSL with patch-level encoder utility |
| Deep learning for abdominal ultrasound: severity of fatty liver | Journal of the Chinese Medical Association, 2021 [web:1040] | Abdominal US disease severity classification | Demonstrates acceptance of abdominal US deep learning in general medical journals | Different disease target; not focal liver lesion differentiation |

## Proposed manuscript contribution

The manuscript should position the contribution in clinical language rather than algorithm novelty language.

### Core contribution statement

A **lightweight self-supervised B-mode ultrasound encoder** is developed for liver lesion analysis, with downstream focus on differentiating HCC from hemangioma and future support for prototype-based clinical interpretation.[UNVERIFIED]

### Specific contributions to emphasize

- A stage-1 pretraining framework is built specifically for grayscale liver ultrasound rather than importing generic natural-image pretraining assumptions.[UNVERIFIED]
- The stage-1 design explicitly seeks to preserve **encoded patch quality** for downstream dense/local tasks while maintaining global invariance.[UNVERIFIED]
- A diversity regularization strategy based on **top-half multi-layer CLS-row diversity plus entropy floor** is introduced to reduce head collapse and encourage complementary regional attention.[UNVERIFIED]
- The final framing is clinically oriented: not merely improving attention-map aesthetics, but supporting robust HCC-vs-hemangioma discrimination and stage-2 HCC prototype induction.[UNVERIFIED]

## Suggested manuscript structure

## 1. Title page

### Draft
- Full title.
- Short running title.
- Author list, affiliations, corresponding author.
- Funding, conflicts, acknowledgments placeholders.

### Checklist to expand
- [ ] Finalize title in clinical rather than overly technical wording.
- [ ] Decide whether to mention “self-supervised” in the title or subtitle only.
- [ ] Add IRB / ethics statement location depending on target journal instructions.

## 2. Abstract

### Draft
This study aims to develop a lightweight self-supervised encoder for B-mode liver ultrasound that improves downstream differentiation between hepatocellular carcinoma and hemangioma while preserving patch-level representation quality for future dense and prototype-based analysis.[UNVERIFIED] Stage-1 pretraining strategies including DINO, DINO+SimMIM, and DINO+SimMIM with diversity regularization are compared, followed by stage-2 downstream evaluation on the HCC-vs-hemangioma task.[UNVERIFIED] The manuscript should report discrimination performance, calibration or clinical utility metrics if available, and qualitative interpretability findings.[UNVERIFIED]

### Checklist to expand
- [ ] Rewrite in structured abstract format required by the journal, typically Purpose / Materials and Methods / Results / Conclusion.[web:1042][web:1044]
- [ ] Add exact dataset size, patient count, study period, and train/validation/test split.
- [ ] Add primary metric and confidence intervals.
- [ ] Add one sentence on clinical implication rather than model novelty.

## 3. Introduction

### Draft
Focal liver lesions are common on abdominal imaging, and distinguishing malignant from benign lesions on grayscale ultrasound remains clinically important yet operator dependent.[web:1042][web:1049] Prior deep learning studies in medical journals have shown promising performance for lesion detection, localization, and classification on abdominal ultrasound, including expert-level comparisons in selected tasks.[web:1042][web:1044][web:1045] Nevertheless, most existing studies emphasize supervised end-task performance, CEUS, or lesion detection rather than the design of a reusable stage-1 representation that preserves local token information for downstream clinical reasoning.[web:1041][web:1042][web:1051]

### Checklist to expand
- [ ] Open with clinical burden and diagnostic uncertainty for HCC vs benign lesions, especially hemangioma.
- [ ] Add why B-mode ultrasound remains important despite CT/MRI availability.
- [ ] Position HCC-vs-hemangioma as clinically meaningful and nontrivial.
- [ ] Explain why a reusable stage-1 encoder matters for limited-label medical datasets.
- [ ] End with a clear study objective and hypothesis.

## 4. Materials and Methods

### 4.1 Study design and cohort

#### Draft
This should be written like a retrospective diagnostic model development study, including institution, study period, inclusion/exclusion criteria, reference standard, and data partitioning strategy, following the style commonly used in abdominal ultrasound AI papers.[web:1042][web:1044]

#### Checklist to expand
- [ ] State retrospective or prospective design.
- [ ] Add IRB approval and informed-consent waiver if applicable.
- [ ] Describe patient selection and exclusion criteria.
- [ ] Clarify whether analysis is image-level, lesion-level, or patient-level.
- [ ] Report the final number of patients, lesions, and images per class.

### 4.2 Ultrasound acquisition and preprocessing

#### Draft
B-mode liver ultrasound images are used, and preprocessing should be described in clinically understandable language, including de-identification, cropping/cleaning of overlays if applicable, resize policy, and augmentation strategy.[UNVERIFIED]

#### Checklist to expand
- [ ] Report scanner vendors, probes, and acquisition variability if known.
- [ ] State whether only still images were used.
- [ ] Explain masking pipeline for SimMIM in simple terms.
- [ ] Separate clinical image preprocessing from model-side augmentation.

### 4.3 Stage-1 self-supervised pretraining

#### Draft
The stage-1 encoder is pretrained using three planned conditions: DINO only, DINO plus SimMIM, and DINO plus SimMIM plus diversity regularization.[UNVERIFIED] DINO provides global representation consistency across views, SimMIM supports patch-level reconstruction, and the diversity term regularizes headwise attention patterns across multiple layers to discourage collapse.[UNVERIFIED]

#### Checklist to expand
- [ ] Keep equations minimal in the main text; move detailed loss equations to supplement if needed.
- [ ] Explain each stage-1 component in clinical-English language first, then technical detail.
- [ ] State why SelfPatch was not included in the primary experiment plan.
- [ ] Add exact architecture, input size, patch size, optimizer, schedule, epochs, and hardware.

### 4.4 Diversity regularization

#### Draft
The diversity module is computed on CLS-to-patch attention patterns from the top half of transformer layers rather than only from the final layer.[UNVERIFIED] A complementary entropy-floor term is added to reduce low-entropy collapse and encourage broader yet still discriminative allocation of headwise attention.[UNVERIFIED]

#### Checklist to expand
- [ ] Explain why this is needed in ultrasound, using the concept of repeated fixation on hyperechoic shortcuts.
- [ ] Add a simple figure or supplement diagram showing CLS-row extraction.
- [ ] Decide whether to keep formulae in the main manuscript or supplement.
- [ ] Clarify the selected default hyperparameters and rationale.

### 4.5 Stage-2 downstream task

#### Draft
After stage-1 pretraining, the encoder is transferred to the downstream HCC-vs-hemangioma classification task.[UNVERIFIED] If prototype induction is included in the submitted version, it should be presented as a clinically motivated interpretability component rather than as a purely algorithmic module.[UNVERIFIED]

#### Checklist to expand
- [ ] Define the stage-2 training and evaluation protocol.
- [ ] Clarify whether fine-tuning is full, partial, or frozen-encoder based.
- [ ] Add class imbalance handling and threshold selection.
- [ ] Decide whether prototype results are in main paper or supplement.

### 4.6 Outcomes and statistical analysis

#### Draft
Primary and secondary outcomes should be reported in medical-journal style, including AUROC, sensitivity, specificity, PPV, NPV, and confidence intervals where appropriate.[web:1042][web:1044] If patient-level clustering exists, patient-level resampling or bootstrap confidence intervals should be considered, as done in prior abdominal ultrasound AI studies.[web:1042]

#### Checklist to expand
- [ ] Define the primary endpoint before listing many metrics.
- [ ] Use patient-level split and patient-level CI if possible.
- [ ] Plan subgroup analyses if lesion size or scanner vendor information is available.
- [ ] Add model comparison method and correction for multiple comparisons if applicable.

## 5. Results

### 5.1 Cohort characteristics

#### Draft
Begin with a table of cohort characteristics and class distribution rather than jumping directly to model performance, in keeping with medical-journal conventions.[web:1042][web:1044]

#### Checklist to expand
- [ ] Create a STARD-style flow diagram for inclusion/exclusion.
- [ ] Prepare Table 1 with demographics and imaging counts.
- [ ] Report any important imbalance or vendor differences.

### 5.2 Main downstream performance

#### Draft
Report stage-2 diagnostic performance across the three stage-1 conditions, emphasizing whether DINO+SimMIM improves downstream discrimination and whether the added diversity term further improves robustness, sensitivity/specificity balance, or interpretability.[UNVERIFIED]

#### Checklist to expand
- [ ] Present AUROC plus clinically interpretable threshold metrics.
- [ ] Include confidence intervals and statistical comparisons.
- [ ] Add patient-level confusion matrices or calibration if available.
- [ ] Keep the focus on clinical diagnostic usefulness, not only internal validation score gains.

### 5.3 Representation-focused analyses

#### Draft
Representation analyses should support the clinical claim that stage-1 pretraining matters.[UNVERIFIED] Candidate analyses include encoded-patch quality, attention-map diversity behavior, late-collapse behavior, and qualitative examples showing whether heads attend to complementary lesion and background regions.[UNVERIFIED]

#### Checklist to expand
- [ ] Show that diversity does not merely make attention prettier.
- [ ] Link representation behavior to downstream benefit.
- [ ] Decide which analyses belong in the supplement to avoid overloading the main paper.

### 5.4 Qualitative interpretation

#### Draft
Include representative ultrasound examples with attention or prototype visualizations, but describe them conservatively as supportive interpretability evidence rather than mechanistic proof.[UNVERIFIED]

#### Checklist to expand
- [ ] Show true-positive, false-positive, and failure cases.
- [ ] Include examples from both HCC and hemangioma.
- [ ] Ask whether lesion boundary, posterior enhancement, or surrounding liver texture seems clinically plausible.

## 6. Discussion

### Draft
The discussion should begin with the main clinical finding: whether a lightweight self-supervised ultrasound encoder improved downstream HCC-vs-hemangioma diagnosis and whether the diversity-regularized stage-1 strategy produced a more usable representation for local reasoning.[UNVERIFIED] It should then compare the work with prior medical-journal literature on abdominal ultrasound AI, emphasizing that this study contributes a representation-learning perspective rather than only another supervised classifier.[web:1042][web:1044][web:1045]

### Checklist to expand
- [ ] First paragraph: summarize main findings only.
- [ ] Second paragraph: compare with prior B-mode and CEUS papers.
- [ ] Third paragraph: explain why patch-quality preservation matters clinically.
- [ ] Fourth paragraph: discuss interpretability carefully without overclaiming.
- [ ] Add explicit limitations: retrospective design, single-center risk, still-image setting, label/reference limitations, and external validation need.
- [ ] End with clinical implications and future prospective validation.

## 7. Conclusion

### Draft
A clinically oriented conclusion should state that stage-1 self-supervised design may improve downstream liver ultrasound diagnosis by producing a reusable representation that balances global discrimination, local patch quality, and stable attention specialization.[UNVERIFIED]

### Checklist to expand
- [ ] Keep to 2–3 sentences.
- [ ] Avoid claiming clinical deployment readiness unless externally validated.
- [ ] Mention future multicenter prospective validation.

## 8. Figures and tables plan

### Draft
The paper should resemble a medical-imaging journal article with a modest number of high-yield figures and tables, rather than a benchmark-heavy ML paper.[web:1042][web:1044]

### Checklist to expand
- [ ] Figure 1: study flow / cohort selection.
- [ ] Figure 2: overview of stage-1 and stage-2 pipeline in medical-style schematic.
- [ ] Figure 3: representative HCC and hemangioma cases with attention/prototype overlays.
- [ ] Table 1: cohort characteristics.
- [ ] Table 2: main performance comparison across three stage-1 plans.
- [ ] Table 3: ablation / representation-support analysis, possibly supplement.

## 9. Writing style rules for this manuscript

### Draft
The manuscript should read like a **clinical diagnostic study with AI methods**, not like an algorithm paper. The Methods section can contain technical detail, but the Introduction, Results, and Discussion should prioritize clinical rationale, patient/sample description, diagnostic endpoints, and practical significance.[web:1042][web:1044][web:1041]

### Checklist to expand
- [ ] Avoid framing the main contribution as “novel architecture” unless truly justified.
- [ ] Prefer “study cohort”, “reference standard”, “diagnostic performance”, and “clinical implication” wording.
- [ ] Keep heavy implementation details in supplement if target journal is clinically oriented.
- [ ] Ensure reporting follows STARD/TRIPOD-AI/CLAIM spirit where applicable.[web:1042]

## 10. Section-by-section TODO summary

- [ ] Finalize journal shortlist and adapt abstract length / figure count / reference style.
- [ ] Freeze the exact primary endpoint.
- [ ] Decide the minimum ablation set for the main manuscript.
- [ ] Prepare a patient-level results table.
- [ ] Prepare failure-case analysis.
- [ ] Decide whether prototype induction is in the main paper or future work.
- [ ] Add ethics, data availability, and code availability statements.
