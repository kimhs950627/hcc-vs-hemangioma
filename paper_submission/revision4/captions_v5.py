"""Figure captions shared by tables_v5.docx and figures_v5.docx.

Single source so the two documents cannot drift apart. Each entry is
(title, description, is_revised) where is_revised marks text changed in this
revision (rendered blue in both documents).
"""

CAPTIONS = [
    ("Figure 1. t-SNE Embedding Space Visualization (Primary Model: EfficientNetV2B0 + CE+SupCon).",
     "Embedding distribution of training and validation sets showing clear intra-class compactness and inter-class separation between HCC (red) and hemangioma (teal) clusters. Left panel illustrates single mean prototype centres; right panel depicts k-means cluster prototypes (k=4).",
     True),
    ("Figure 2. Confusion Matrices of Primary Model on Held-out Test Set (n=268).",
     "Test set only. (A) Confidence score at the prespecified 0.50 operating point: 134 TP, 0 FP, 6 FN, 128 TN. Panels A-C are presented together in a single figure. (B) HCC cosine similarity score and (C) Δscore at their validation Youden operating points (exploratory): 140 TP, 1 FP, 0 FN, 127 TN. The single false positive is the sample that is classified correctly at 0.50, so the failure modes differ by operating point (Table 3A, 3B).",
     True),
    ("Figure 3. Triple Receiver Operating Characteristic (ROC) Curves on Test Set.",
     "Test set only. ROC curves for ROC-A (confidence score), ROC-B (HCC cosine similarity score) and ROC-C (Δscore). All three reach AUROC 1.000 and the curves overlap, so DeLong testing carries no information; no superiority or non-inferiority is claimed. The validation panel is not shown (test-only reporting).",
     True),
    ("Figure 4. HCC Cosine Score Distributions with and without Supervised Contrastive Learning (SupCon) in True HCC Cases (n=417; Validation 277 + Test 140).",
     "The same held-out HCC images were scored by the CE-only and CE+SupCon models (HCC cosine similarity score to the training-set mean HCC prototype). (A) Density distribution, EfficientNetV2B0. (B) Paired individual shift, EfficientNetV2B0: mean 0.783 ± 0.256 (median 0.882) to 0.908 ± 0.160 (median 0.970); mean paired difference +0.125 (95% CI 0.110–0.140); paired t = 16.94; Wilcoxon signed-rank P < 0.001; score increased in 94.7% of images; variance 0.065 to 0.026 (−60.7%; Pitman-Morgan t = 16.84, P < 0.001). (C) Density distribution, ResNet50V2. (D) Paired individual shift, ResNet50V2: mean 0.774 ± 0.265 to 0.878 ± 0.230; mean paired difference +0.104; paired t = 10.98; Wilcoxon P < 0.001; variance 0.070 to 0.053 (−24.3%; Pitman-Morgan t = 4.06, P < 0.001). CE, cross-entropy.",
     True),
    ("Figure 5. Hemangioma Cosine Score Distributions with and without Supervised Contrastive Learning (SupCon) in True Hemangioma Cases (n=381; Validation 253 + Test 128).",
     "The same held-out hemangioma images were scored by the CE-only and CE+SupCon models (hemangioma cosine similarity score to the training-set mean hemangioma prototype). (A) Density distribution, EfficientNetV2B0. (B) Paired individual shift, EfficientNetV2B0: mean 0.937 ± 0.050 (median 0.955) to 0.985 ± 0.006 (median 0.985); mean paired difference +0.048 (95% CI 0.043–0.052); paired t = 19.71; Wilcoxon signed-rank P < 0.001; variance 0.00253 to 0.00003 (Pitman-Morgan t = 106.23, P < 0.001). (C) Density distribution, ResNet50V2. (D) Paired individual shift, ResNet50V2: mean 0.966 ± 0.019 to 0.990 ± 0.004; mean paired difference +0.024 (95% CI 0.022–0.026); paired t = 26.54; Wilcoxon P < 0.001; variance 0.00036 to 0.00002 (Pitman-Morgan t = 49.18, P < 0.001). The large relative variance reduction partly reflects the already small baseline variance under CE-only training.",
     True)
]
