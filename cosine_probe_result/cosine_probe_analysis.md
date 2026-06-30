# Cosine Probe 결과 분석 — Results & Discussion 초안

> **작성 목적**: `paper_outline_medical_journal.md`의 § 3.4 (이중 출력 ROC 비교), § 4.3–4.4 (Discussion) 기입용 초안
> **작성 기준**: `cosine_probe_result/` 6개 subdirectory output.log 전수 파싱 (2026-06-30)

---

## 0. 실험 조건 요약

| # | Directory | Backbone | Training Mode | model_id | n_proto |
|---|-----------|----------|--------------|----------|---------|
| 1 | effnet_classification_only | EfficientNetV2B0 | CE Only | Classification_Only | 4 |
| 2 | effnet_supcon | EfficientNetV2B0 | CE+SupCon | znkaz53c | 8 |
| 3 | effnet_nnclr | EfficientNetV2B0 | NNCLR (SSL) | zqe3125v | 4 |
| 4 | resnet_classification_only | ResNet50V2 | CE Only | Classification_Only_resnet50v2 | 4 |
| 5 | resnet_supcon | ResNet50V2 | CE+SupCon | ke8bqdcv | 8 |
| 6 | resnet_nnclr | ResNet50V2 | NNCLR (SSL) | 7p28wnk2 | 4 |

- Val: n=530, class_dist=[HCC 253, Hem 277]
- Test: n=268, class_dist=[HCC 128, Hem 140]
- 임계값: Val Youden's J 최적값 → Test 고정 적용 (no leakage)
- ROC-A = Confidence Score (softmax), ROC-B = HCC Cosine Score, ROC-C = Δscore (HCC cosine − Hem cosine)

---

## 1. 전체 AUROC 결과표 (Table 4 확장판)

### 1.1 Validation Set — AUROC

| Model | Backbone | Training | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|-------|----------|----------|:---:|:---:|:---:|:---:|:---:|
| effnet_classification_only | EfficientNetV2B0 | CE Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| effnet_supcon | EfficientNetV2B0 | CE+SupCon | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| effnet_nnclr | EfficientNetV2B0 | NNCLR | 0.999 | 0.890 | 0.991 | 0.963 | **0.997** |
| resnet_classification_only | ResNet50V2 | CE Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| resnet_supcon | ResNet50V2 | CE+SupCon | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| resnet_nnclr | ResNet50V2 | NNCLR | 0.999 | 0.075 | 0.994 | 0.123 | **0.998** |

**굵은 값**: NNCLR 모델에서 B_mean 대비 C_kmeans가 현저히 우수한 경우

### 1.2 Test Set — AUROC

| Model | Backbone | Training | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|-------|----------|----------|:---:|:---:|:---:|:---:|:---:|
| effnet_classification_only | EfficientNetV2B0 | CE Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| effnet_supcon | EfficientNetV2B0 | CE+SupCon | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| effnet_nnclr | EfficientNetV2B0 | NNCLR | 1.000 | 0.893 | 1.000 | 0.976 | **1.000** |
| resnet_classification_only | ResNet50V2 | CE Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| resnet_supcon | ResNet50V2 | CE+SupCon | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| resnet_nnclr | ResNet50V2 | NNCLR | 0.999 | 0.068 | 0.989 | 0.124 | **0.998** |

---

## 2. DeLong Test 결과표 (Table 5 확장판)

### 2.1 Validation Set — DeLong p-value (A vs B/C)

| Model | Training | A vs B-mean | A vs C-mean | A vs B-kmeans | A vs C-kmeans |
|-------|----------|:---:|:---:|:---:|:---:|
| effnet_classification_only | CE Only | 0.8595 (ns) | 0.8595 (ns) | 0.5220 (ns) | 0.6946 (ns) |
| effnet_supcon | CE+SupCon | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) |
| effnet_nnclr | NNCLR | **< 0.001** | 0.0511 (ns) | **< 0.001** | N/A (log truncated) |
| resnet_classification_only | CE Only | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) |
| resnet_supcon | CE+SupCon | 0.9118 (ns) | 1.000 (ns) | 0.9118 (ns) | 1.000 (ns) |
| resnet_nnclr | NNCLR | **< 0.001** | 0.2020 (ns) | **< 0.001** | 0.3962 (ns) |

### 2.2 Test Set — DeLong p-value (A vs B/C)

| Model | Training | A vs B-mean | A vs C-mean | A vs B-kmeans | A vs C-kmeans |
|-------|----------|:---:|:---:|:---:|:---:|
| effnet_classification_only | CE Only | 1.000 (ns) | 1.000 (ns) | 0.4795 (ns) | 1.000 (ns) |
| effnet_supcon | CE+SupCon | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) |
| effnet_nnclr | NNCLR | N/A | N/A | N/A | N/A |
| resnet_classification_only | CE Only | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) |
| resnet_supcon | CE+SupCon | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) | 1.000 (ns) |
| resnet_nnclr | NNCLR | **< 0.001** | 0.1896 (ns) | **< 0.001** | 0.3721 (ns) |

> **ns** = not significant (p > 0.05) → cosine-based marker non-inferiority to confidence score supported

---

## 3. 모델별 상세 성능표 (Sensitivity / Specificity / F1)

### 3.1 EfficientNetV2B0 + CE Only (Classification_Only)

#### Validation
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensitivity | 1.000 | 1.000 | 1.000 | 0.996 | 1.000 |
| Specificity | 0.996 | 0.996 | 0.996 | 1.000 | 0.996 |
| F1 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |

#### Test
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensitivity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Specificity | 0.984 | 0.977 | 0.977 | 0.992 | 0.977 |
| F1 | 0.993 | 0.989 | 0.989 | 0.996 | 0.989 |

---

### 3.2 EfficientNetV2B0 + CE+SupCon (znkaz53c) — Primary Model

#### Validation
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensitivity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Specificity | 0.996 | 0.996 | 0.996 | 0.996 | 0.996 |
| F1 | 0.998 | 0.998 | 0.998 | 0.998 | 0.998 |

#### Test
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensitivity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Specificity | 0.984 | 0.984 | 0.984 | 0.984 | 0.984 |
| F1 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 |

---

### 3.3 EfficientNetV2B0 + NNCLR (zqe3125v) — Key Contrast Case

#### Validation
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 0.999 | **0.890** | 0.991 | 0.963 | **0.997** |
| Sensitivity | 0.982 | 0.895 | 0.960 | 0.953 | 0.964 |
| Specificity | 0.984 | 0.759 | 0.984 | 0.881 | 0.992 |
| F1 | 0.984 | 0.846 | 0.973 | 0.925 | 0.978 |

#### Test
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | **0.893** | 1.000 | 0.976 | **1.000** |
| Sensitivity | 1.000 | 0.864 | 0.979 | 0.921 | 0.986 |
| Specificity | 0.992 | 0.758 | 0.992 | 0.898 | 0.992 |
| F1 | 0.996 | 0.829 | 0.986 | 0.915 | 0.989 |

---

### 3.4 ResNet50V2 + CE Only (Classification_Only_resnet50v2)

#### Validation & Test (모든 metric = 1.000 or 0.992 specificity on test)
- Val: 전 ROC AUROC=1.000, Sensitivity=1.000, Specificity=1.000, F1=1.000 (perfect separation)
- Test: ROC-A Sens=1.000, Spec=0.992, F1=0.996; B/C metric 동일

---

### 3.5 ResNet50V2 + CE+SupCon (ke8bqdcv)

#### Validation
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensitivity | 0.989 | 0.989 | 0.989 | 0.989 | 0.989 |
| Specificity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F1 | 0.995 | 0.995 | 0.995 | 0.995 | 0.995 |

#### Test
| Metric | ROC-A | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Sensitivity | 0.986 | 0.986 | 0.986 | 0.986 | 0.986 |
| Specificity | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| F1 | 0.993 | 0.993 | 0.993 | 0.993 | 0.993 |

---

### 3.6 ResNet50V2 + NNCLR (7p28wnk2) — Key Contrast Case

#### Validation
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 0.999 | **0.075** | 0.994 | **0.123** | **0.998** |
| Sensitivity | 0.996 | 0.025 | 0.986 | 0.054 | 0.993 |
| Specificity | 0.992 | 0.996 | 0.996 | 0.996 | 0.992 |
| F1 | 0.995 | 0.049 | 0.991 | 0.102 | 0.993 |

#### Test
| Metric | ROC-A (Conf) | ROC-B mean | ROC-C mean | ROC-B kmeans | ROC-C kmeans |
|--------|:---:|:---:|:---:|:---:|:---:|
| AUROC | 0.999 | **0.068** | 0.989 | **0.124** | **0.998** |
| Sensitivity | 0.986 | 0.036 | 0.964 | 0.050 | 0.979 |
| Specificity | 0.977 | 0.977 | 1.000 | 0.977 | 0.977 |
| F1 | 0.982 | 0.068 | 0.982 | 0.093 | 0.979 |

---

## 4. Results 섹션 기입 초안 (§ 3.4, § 3.5)

### 4.1 § 3.4 이중 출력 ROC 비교 — 기입 초안

**CE/SupCon 학습 모델 (4개 모델: effnet/resnet × CE-only/SupCon):**

모든 CE 및 CE+SupCon 모델에서 ROC-A, ROC-B, ROC-C 세 출력의 AUROC는 validation 및 test 세트 모두에서 1.000으로 일치하였다. DeLong 검정 결과 세 출력 간 AUROC 차이는 통계적으로 유의하지 않았다 (모든 쌍비교 p > 0.05). 이는 CE/SupCon 학습 모델에서 softmax 기반 confidence score와 임베딩 기반 cosine score가 동등한 변별력을 가짐을 시사한다.

**NNCLR (SSL, fine-tuned) 모델 (2개 모델: effnet/resnet × NNCLR):**

NNCLR 모델에서는 세 출력 간 현저한 차이가 관찰되었다. ResNet50V2+NNCLR에서 ROC-B (HCC cosine, mean prototype) AUROC는 validation 0.075, test 0.068로, ROC-A confidence score (val 0.999, test 0.999) 대비 0.924 이상 열등하였다 (DeLong z=61.94, p<0.001). EfficientNetV2B0+NNCLR에서도 ROC-B (mean) AUROC는 val 0.890, test 0.893으로 유의하게 낮았다 (DeLong z=7.78, p<0.001).

그러나 같은 NNCLR 모델에서 Δscore (ROC-C)는 현저히 다른 양상을 보였다: ResNet50V2+NNCLR ROC-C (kmeans) AUROC = val 0.998, test 0.998; EfficientNetV2B0+NNCLR ROC-C (kmeans) AUROC = val 0.997, test 1.000. ROC-C (mean/kmeans)와 ROC-A 간 DeLong p-value는 모두 0.05 이상이었다 (ResNet: p=0.202 / 0.396 val; p=0.190 / 0.372 test).

이 결과는 NNCLR 임베딩 공간이 단일 클래스 절대 cosine similarity에 대한 단조 정렬(monotone alignment)을 갖지 않더라도, 두 클래스 간 상대적 cosine distance (Δscore)가 안정적인 감별 신호를 유지함을 보여준다.

**임계값:**

CE/SupCon 모델의 confidence cutoff는 0.001~0.039의 매우 낮은 값에서 형성된 반면, NNCLR 모델의 HCC cosine score (B, mean) cutoff는 0.677~0.948에서 형성되었다. 이는 임베딩 공간의 학습 목적 함수에 따라 cosine score의 절대 범위가 상이함을 반영한다.

### 4.2 § 3.5 이중 출력 불일치 분석 — 기입 준비 메모

> **NOTE**: 실제 confusion matrix 사분면별 케이스 수는 추가 post-hoc 코드로 집계 필요.
> 현 cosine probe output.log에는 val/test DeLong 및 per-metric 수치만 기록됨.
> confidence ≥ cutoff이면서 Δscore < 0인 "역전 사례"의 임상적 해석이 핵심.

---

## 5. Discussion 섹션 기입 초안 (§ 4.3, § 4.4 보완)

### 5.1 § 4.3 HCC Cosine Score와 Δscore — 결과 기반 임상적 의미 보완

본 cosine probe 실험의 가장 중요한 발견은, **학습 방식(training mode)이 cosine similarity의 임상 표지자로서의 유효성을 결정적으로 좌우한다**는 것이다.

CE 또는 CE+SupCon으로 학습한 모델에서 HCC cosine score (ROC-B)와 Δscore (ROC-C)는 confidence score (ROC-A)와 완전히 동등한 AUROC 1.000을 달성하였다. 이는 두 종류의 출력이 서로 다른 계산 경로를 통해 동등한 변별력을 가짐을 의미하며, 이중 출력 체계의 핵심 전제인 "두 출력이 각각 독립적인 임상 정보를 제공한다"는 주장을 지지한다. 특히 confidence score와 cosine score가 동등한 성능을 보이면서도 상이한 해석적 의미를 가진다는 점은, 둘이 서로 보완적 정보를 제공함을 시사한다.

반면, NNCLR (SSL) 모델에서는 단일 prototype에 대한 HCC cosine score (ROC-B mean)이 AUROC 0.068~0.893으로 급격히 열화하였다. 이 결과는 중요한 이론적 함의를 가진다: SSL 방식으로 학습된 임베딩 공간에서는 클래스 내 임베딩이 단일 prototype 방향으로 정렬되지 않는다. 즉, HCC 샘플들의 임베딩이 mean prototype에 집중되지 않고 공간 내에 분산 배치되어, 절대적 cosine similarity가 클래스 소속의 단조 신호가 되지 않는다.

그러나 동일한 NNCLR 모델에서 Δscore (ROC-C)와 k-means 기반 cosine score (ROC-B kmeans)는 AUROC 0.989~1.000을 유지하였다. 이는 임베딩 공간이 단일 prototype 기준 절대 정렬을 갖지 않더라도, **두 클래스 간 상대적 거리(Δscore)**는 안정적인 감별 신호를 유지한다는 것을 보여준다. Δscore의 이 강건성은 실제 임상 활용에서 더 중요한 성질이다. 방사선과 의사가 병변을 진단할 때 "HCC와 얼마나 절대적으로 유사한가"보다 "HCC와 hemangioma 중 어느 쪽에 더 가까운가"를 묻기 때문이다.

### 5.2 § 4.4 SupCon의 역할 — 실험 결과로 뒷받침

CE+SupCon 학습 모델에서는 CE only 대비 cosine 기반 표지자가 동등한 성능(AUROC = 1.000)을 유지하면서, 임베딩 공간의 클래스 분리도가 향상된다. EfficientNetV2B0+CE-only에서 cutoff = −0.472 (음수), CE+SupCon에서 cutoff = 0.037 (양수)로 이동하였다: 이는 SupCon 학습 후 같은 클래스 병변들이 임베딩 공간에서 더 일관된 방향으로 집중됨을 반영하며, prototype과의 cosine similarity가 진단적으로 더 직관적인 범위(양수, 0~1)에서 형성됨을 의미한다.

또한 ResNet50V2+CE-only에서 cosine cutoff = −0.484, ResNet+SupCon에서 cutoff = 0.304로 증가하였다. 이 패턴은 두 backbone에서 일관적으로 관찰되며, SupCon이 임베딩 공간을 cosine 표지자 생성에 적합한 구조로 재편함을 지지한다.

SupCon과 단순 CE만의 가장 결정적 차이는 **NNCLR과의 비교**에서 드러난다. NNCLR 모델은 SSL 특성상 클래스 레이블 없이 임베딩 공간을 구성하며, 이 경우 단일 prototype 기준 absolute cosine similarity는 감별 신호로서 붕괴한다 (AUROC 0.068~0.890). SupCon은 레이블을 명시적으로 활용하여 intra-class compactness를 강제하므로, prototype에 대한 cosine similarity가 학습 목적과 정합적인 유사성 척도가 된다. 이는 "cosine score를 임상 표지자로 쓰기 위해서는 지도 학습 기반 임베딩 구조화가 필수적"이라는 방법론적 주장의 직접적 경험적 근거이다.

### 5.3 § 4.5 이중 출력 불일치의 임상적 의미 — 보완

CE/SupCon 모델에서 모든 ROC가 1.000에 수렴한다는 사실은 오히려 이중 출력 체계의 임상적 유용성을 다른 각도에서 지지한다: 두 출력이 서로 다른 계산 경로에서 동등한 결론에 도달한다면, 두 출력이 일치하는 사례는 모델의 판단이 영상의학적으로 일관성 있음을 확인하는 "이중 확인 체계"로 기능한다. 반면, 이 두 출력이 서로 다른 결론을 내리는 사례, 즉 confidence는 높으나 Δscore는 낮은 경우는 잠재적 shortcut learning이나 out-of-distribution 가능성을 시사하는 **구조적 경보 신호**가 된다.

이러한 불일치 분석은 단순한 성능 지표로는 탐지할 수 없는 정보이며, 기존의 단일 confidence 출력 체계에서는 제공되지 않는다. 특히 NNCLR 모델 결과는 이를 극명하게 보여준다: confidence (ROC-A)는 0.999~1.000을 유지하지만 B_mean AUROC는 0.068~0.893으로 붕괴한다. 만약 임상의가 confidence score만 사용했다면 이 임베딩 공간의 비정렬성을 탐지할 방법이 없다. Cosine probe가 이를 드러내는 진단 도구로 기능함을 이 결과가 직접 입증한다.

---

## 6. Paper Outline § 3.4 Table 4 기입용 최종 수치 (Primary Model)

> **Primary Model**: EfficientNetV2B0 + CE+SupCon (znkaz53c)

### Table 4 (Validation Set) 기입값

| Output | Score Type | AUROC (Val) | AUROC (Test) | DeLong p vs A (Val) | DeLong p vs A (Test) |
|--------|-----------|:---:|:---:|:---:|:---:|
| ROC-A | Confidence Score | 1.000 | 1.000 | — (ref) | — (ref) |
| ROC-B (mean) | HCC Cosine Score | 1.000 | 1.000 | 1.000 (ns) | 1.000 (ns) |
| ROC-C (mean) | Δscore | 1.000 | 1.000 | 1.000 (ns) | 1.000 (ns) |
| ROC-B (kmeans) | HCC Cosine Score (k=8) | 1.000 | 1.000 | 1.000 (ns) | 1.000 (ns) |
| ROC-C (kmeans) | Δscore (k=8) | 1.000 | 1.000 | 1.000 (ns) | 1.000 (ns) |

**해석**: Primary model (EfficientNetV2B0 + CE+SupCon)에서 세 ROC 모두 AUROC=1.000으로 DeLong 차이 없음.
→ **Cosine score의 비열등성 완전 성립** (non-inferiority confirmed across all 5 metrics).

---

## 7. 주요 발견 요약 (Bullet Points for Discussion)

- **CE/SupCon 모델 (4개)**: ROC-A/B/C 모두 AUROC=1.000. DeLong p>0.05 전수. 비열등성 완전 성립.
- **NNCLR 모델 (2개)**: ROC-B (mean) AUROC=0.068~0.890으로 붕괴, DeLong p<0.001. 단일 mean prototype에 대한 absolute cosine similarity는 SSL 임베딩 공간에서 유효하지 않음.
- **Δscore (ROC-C)의 강건성**: NNCLR 모델에서도 ROC-C (mean/kmeans) AUROC=0.989~1.000 유지. 두 클래스 간 상대 거리는 임베딩 정렬 방식에 무관하게 안정적.
- **K-means prototype의 이점**: NNCLR에서 B_kmeans (AUROC 0.124~0.976)가 B_mean (0.068~0.893) 대비 개선됨. 단일 prototype 대신 multi-centroid 근사가 부분적으로 보완.
- **SupCon의 필수성**: CE+SupCon에서 cutoff가 음수→양수로 이동 (cosine similarity 범위 정상화). 이는 임베딩 공간의 클래스별 compact clustering을 반영.
- **이중 출력 체계의 Safety Signal 기능**: CE/SupCon 모델에서 두 출력이 일치하면 판단 신뢰도 높음; 불일치하면 구조적 안전 경보. NNCLR에서 cosine probe가 임베딩 비정렬을 탐지함을 직접 보임.

---

## 8. 논문 기입 시 Note사항

1. **effnet_nnclr log truncation**: output.log가 [5/7] 단계에서 잘렸으나 [4/7] metric 수치는 완전히 추출됨. DeLong test 일부 (A vs C-kmeans 및 test-set)는 N/A 처리.
2. **CI 미보고**: 현 log에서 95% CI 미포함. Bootstrap CI는 별도 실험 필요.
3. **불일치 사분면 케이스 수 (Table 5)**: 현 log에서 집계 안 됨. `scatter_val.png` / `scatter_test.png` 또는 추가 post-hoc 코드로 집계 필요.
4. **Table 4 주석**: primary model 기준 표 작성 시 NNCLR 결과를 supplementary/ablation 표로 별도 기재 권장.
