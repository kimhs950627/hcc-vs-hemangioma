# Recomputed Metrics and Tables for Revision

## Table 2. Backbone × Training Mode (Validation Set, Confidence Score)
|   # | Backbone         | Training Mode   |      Cutoff |   AUROC |   Sensitivity (%) |   Specificity (%) |   F1 Score | TP/FP/FN/TN   |
|----:|:-----------------|:----------------|------------:|--------:|------------------:|------------------:|-----------:|:--------------|
|   1 | ResNet50V2       | CE Only         | 0.000182887 |  1      |            100    |            100    |     1      | 277/0/0/253   |
|   2 | EfficientNetV2B0 | CE Only         | 0.00164298  |  0.9999 |            100    |             99.6  |     0.9982 | 277/1/0/252   |
|   3 | ResNet50V2       | CE + SupCon     | 0.000202953 |  1      |             99.64 |            100    |     0.9982 | 276/0/1/253   |
|   4 | EfficientNetV2B0 | CE + SupCon     | 0.00108838  |  1      |            100    |            100    |     1      | 277/0/0/253   |
|   5 | ResNet50V2       | NNCLR (SSL)     | 0.000335896 |  0.9994 |             99.64 |             99.21 |     0.9946 | 276/2/1/251   |
|   6 | EfficientNetV2B0 | NNCLR (SSL)     | 0.00380852  |  0.9986 |             98.19 |             98.42 |     0.9837 | 272/4/5/249   |

## Table 3. Primary Model — Primary Operating Point (Confidence Score ≥ 0.50), Test Set
Prespecified conventional operating point. Test-only reporting; thresholds were derived on validation.

| Metric                   | Test Set (n=268)      |
|:-------------------------|:----------------------|
| AUROC                    | 1.000                 |
| Accuracy (%) (95% CI)    | 97.76 (95.18–98.96)   |
| Sensitivity (%) (95% CI) | 95.71 (90.98–98.00)   |
| Specificity (%) (95% CI) | 100.00 (97.10–100.00) |
| PPV (%)                  | 100.00                |
| NPV (%)                  | 95.52                 |
| F1 Score                 | 0.9781                |
| TP / FP / FN / TN        | 134 / 0 / 6 / 128     |

*95% CI = Wilson score interval (image level).*

## Table 3B. Similarity Metrics — Exploratory Operating Points, Test Set

| Output            | Cutoff (Val-derived) | Test equivalence interval | Test Sens/Spec (%) | Test Acc (%) | Test TP/FP/FN/TN  |
|:------------------|:---------------------|:--------------------------|:-------------------|:-------------|:------------------|
| HCC cosine (mean) | −0.004843            | (0.0325, 0.2583]          | 100.00 / 99.22     | 99.63        | 140 / 1 / 0 / 127 |
| Δscore (mean)     | −0.958471            | (−0.8879, −0.5435]        | 100.00 / 99.22     | 99.63        | 140 / 1 / 0 / 127 |

*Cutoffs derived on validation (Youden); performance, confusion matrices and equivalence intervals reported for the test set only.
Discordance vs the 0.50 operating point: 6 vs 1; McNemar exact p = 0.125. No superiority or non-inferiority claim.*

## Table 4. Dual-Output Performance Comparison (Primary Model), Test Set
| Output   | Score Type       | Prototype     |   Cutoff |   AUROC (Test) |   Sens (%) |   Spec (%) | Errors (FP/FN)   |
|:---------|:-----------------|:--------------|---------:|---------------:|-----------:|-----------:|:-----------------|
| ROC-A    | Confidence Score | —             |     0.50 |              1 |      95.71 |     100.00 | 0/6              |
| ROC-B    | HCC Cosine Score | mean          |  -0.0048 |              1 |     100.00 |      99.22 | 1/0              |
| ROC-C    | Δscore           | mean          |  -0.9585 |              1 |     100.00 |      99.22 | 1/0              |
| ROC-B    | HCC Cosine Score | k-means (k=4) |   0.0288 |              1 |     100.00 |      99.22 | 1/0              |
| ROC-C    | Δscore           | k-means (k=4) |  -0.9265 |              1 |     100.00 |      99.22 | 1/0              |

*Test-set only. ROC-A uses the prespecified 0.50 operating point; ROC-B/ROC-C use validation Youden points (exploratory). At the similarity operating points all outputs misclassify the same single image (index 576, hemangioma → HCC, FP=1, FN=0); at 0.50 the errors are false negatives instead (FP=0, FN=6). DeLong is uninformative at AUROC 1.000 — no non-inferiority claim.*

