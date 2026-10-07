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

## Table 3. Primary Model Confidence Score Performance (EfficientNetV2B0 + CE + SupCon)
Operating Cutoff (Youden J on Val): 0.001088

| Metric                   | Validation Set        | Test Set              |
|:-------------------------|:----------------------|:----------------------|
| AUROC                    | 1.000                 | 1.000                 |
| Accuracy (%) (95% CI)    | 100.00 (99.31–100.00) | 99.63 (97.94–99.99)   |
| Sensitivity (%) (95% CI) | 100.00 (98.68–100.00) | 100.00 (97.40–100.00) |
| Specificity (%) (95% CI) | 100.00 (98.55–100.00) | 99.22 (95.72–99.98)   |
| PPV (%)                  | 100.00                | 99.29                 |
| NPV (%)                  | 100.00                | 100.00                |
| F1 Score                 | 1.0000                | 0.9964                |
| TP / FP / FN / TN        | 277 / 0 / 0 / 253     | 140 / 1 / 0 / 127     |

## Table 4. Dual-Output Performance Comparison (Primary Model)
| Output   | Score Type       | Prototype     |   Cutoff (Val) |   AUROC (Val) |   AUROC (Test) |   Test Sens (%) |   Test Spec (%) | Test Errors (FP/FN)   |
|:---------|:-----------------|:--------------|---------------:|--------------:|---------------:|----------------:|----------------:|:----------------------|
| ROC-A    | Confidence Score | —             |         0.0011 |             1 |              1 |             100 |           99.22 | 1/0                   |
| ROC-B    | HCC Cosine Score | mean          |        -0.0048 |             1 |              1 |             100 |           99.22 | 1/0                   |
| ROC-C    | Δscore           | mean          |        -0.9585 |             1 |              1 |             100 |           99.22 | 1/0                   |
| ROC-B    | HCC Cosine Score | k-means (k=4) |         0.0288 |             1 |              1 |             100 |           99.22 | 1/0                   |
| ROC-C    | Δscore           | k-means (k=4) |        -0.9265 |             1 |              1 |             100 |           99.22 | 1/0                   |

*Note: In the test set (n=268), all 5 outputs misclassify the exact same single image (index 576, Hemangioma predicted as HCC, FP=1, FN=0).*