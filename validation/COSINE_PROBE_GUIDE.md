# `validation/cosine_probe.py` — 기능 정리

## 핵심 개념: 왜 Cosine Probe인가?

기존 분류 파이프라인에서 HCC 예측값은 `softmax[:, 1]`로 정의했다.  
그러나 이 값은 **보정된 확률(calibrated probability)이 아니라 confidence score**다.

| 개념 | 정의 | 한계 |
|---|---|---|
| `confidence_score` | softmax output — 모델이 HCC라고 얼마나 확신하는지 | 과확신(overconfidence) 경향, 클래스 간 logit 비율에 불과 |
| `hcc_cosine_score` | query embedding과 HCC prototype bank의 평균 cosine similarity | radiologist의 진단 추론과 alignment — "이 병변이 학습된 HCC와 얼마나 닮았는가" |
| `delta_score` | `hcc_cosine_score − hem_cosine_score` | HCC와 Hemangioma를 동시에 고려한 변별 margin |

Guo et al. (ICML 2017, *On Calibration of Modern Neural Networks*)에 따르면  
현대 딥러닝 모델의 softmax output은 체계적으로 overconfident하며,  
temperature scaling 같은 calibration 없이는 true probability로 해석할 수 없다.

SupCon loss는 cosine similarity가 의미 있도록 embedding space를 구성하는 목적함수이므로,  
SupCon으로 학습한 모델에서 cosine-based score를 사용하는 것은 이론적으로 일관성이 있다.

---

## 전체 기능 목록

### Public API

| 함수 | 설명 |
|---|---|
| `run_cosine_probe()` | 기본 파이프라인 (embedding → prototype bank → score → ROC → DeLong → t-SNE → WandB) |
| `run_cosine_probe_v2()` | 모든 기능 통합 진입점 — `run_cosine_probe()` 결과를 base로 5개 extended 분석 추가 실행 |
| `delong_test()` | DeLong 비모수 AUROC 비교 (단독 호출 가능) |
| `plot_triple_roc()` | Triple ROC 단독 호출 |
| `plot_cosine_distribution()` | Cosine score KDE 분포 단독 호출 |
| `plot_tsne()` | t-SNE 단독 호출 |
| `plot_reliability_diagram()` | Calibration reliability diagram 단독 호출 |
| `plot_conf_vs_cosine_scatter()` | Confidence vs Cosine scatter 단독 호출 |
| `plot_confusion_matrix()` | Confusion matrix 단독 호출 |
| `run_test_delong()` | Test set DeLong 단독 호출 |

---

## 기존 기능 (`run_cosine_probe`)

### [1] Embedding 로드 / 추출

**무엇을 보여주는가**  
모델이 이미지에서 추출한 latent representation(embedding)을 수집한다.  
이 embedding이 cosine similarity 계산의 기반이 된다.

| 파라미터 | 동작 |
|---|---|
| `embedding_source="saved"` | extval에서 저장한 `hcc_rep.npy` / `hemangioma_rep.npy` 우선 로드 (VICReg/ConvHybrid) |
| `embedding_source="live"` | `train_ds`에서 실시간 추출 (SupCon/Benchmark) |
| fallback | saved 로드 실패 시 자동으로 live 추출로 전환 |

`_extract_encoder_repr()` 함수는 `model.encoder`, `model.model.encoder`, 또는 model 자체에서  
CLS token / GAP vector를 자동 탐색한다 — wrapping 없이 SupCon/VICReg/Benchmark 모두 동작.

---

### [2] Prototype Bank 구성

**무엇을 보여주는가**  
"전형적인 HCC / Hemangioma"를 대표하는 벡터 집합을 구성한다.  
radiologist가 학습을 통해 머릿속에 형성한 "전형적 HCC 이미지"의 수학적 표현이다.

두 가지 mode가 **항상 동시에** 계산된다:

| Mode | 방법 | Prototype 수 | 의미 |
|---|---|---|---|
| `mean` | train set HCC/Hem 임베딩의 산술 평균 | 1개 | 클래스 무게중심 — 가장 단순한 "전형" |
| `kmeans` | K-Means (k=`n_proto`, default 8) | k개 | 클래스 내 subtype 다양성 포착 |

저장 파일: `hcc_bank_mean.npy`, `hem_bank_mean.npy`, `hcc_bank_kmeans.npy`, `hem_bank_kmeans.npy`

---

### [3] Score 추출 (val + test)

**무엇을 보여주는가**  
각 샘플에 대해 3가지 score를 계산하고, 이를 CSV로 저장한다.

$$
\text{HCC Cosine Score}(x) = \frac{1}{K} \sum_{k=1}^{K} \cos(\mathbf{z}(x),\, \mathbf{p}_k^{HCC})
$$

$$
\Delta\text{Score}(x) = \text{HCC Cosine Score}(x) - \text{Hem Cosine Score}(x)
$$

| Score | 의미 |
|---|---|
| `confidence_score` | softmax P(HCC) — 모델의 확신 정도 |
| `hcc_cosine_mean/kmeans` | HCC prototype과의 유사도 — 병변이 HCC를 얼마나 닮았는지 |
| `hem_cosine_mean/kmeans` | Hemangioma prototype과의 유사도 |
| `delta_mean/kmeans` | 두 유사도의 차 — HCC vs Hem 변별 margin |

저장 파일: `val_scores.csv`, `test_scores.csv`

---

### [4] Cutoff 결정

**무엇을 보여주는가**  
Val set 기준으로 각 score의 최적 임계값을 결정한다.

| 전략 | 방법 |
|---|---|
| `youden` | Youden Index 최대화: `argmax(TPR − FPR)` |
| `sens_first` | 목표 sensitivity (`sens_target=0.90`) 이상 구간에서 최고 specificity 선택 |

5개 ROC 각각(A, B-mean, B-kmeans, C-mean, C-kmeans)에 대해 독립적으로 cutoff 계산.  
저장 파일: `cutoffs.csv`

---

### [5] Metrics at Cutoff

**무엇을 보여주는가**  
임계값 적용 후 val/test의 임상 성능 지표를 계산한다.

| 지표 | 의미 |
|---|---|
| AUROC | threshold-independent 분류 성능 |
| Sensitivity | HCC를 HCC로 맞힌 비율 (recall) |
| Specificity | Hem을 Hem으로 맞힌 비율 |
| PPV | HCC 예측이 실제 HCC일 비율 |
| NPV | Hem 예측이 실제 Hem일 비율 |
| F1 | Sensitivity와 PPV의 조화평균 |

저장 파일: `val_metrics.csv`, `test_metrics.csv`

---

### [6] Triple ROC Plot (`plot_triple_roc`)

**무엇을 보여주는가**  
ROC-A(confidence), ROC-B(cosine), ROC-C(delta) 5개 커브를 한 그림에 overlay한다.  
세 score가 **서로 다른 정보 축**을 가지는지, 또는 동등한 변별력을 갖는지 한눈에 비교한다.

![Triple ROC 예시 구조]

- **ROC-A** (파란 점선): softmax 기반 기존 baseline
- **ROC-B mean** (금색 실선): 평균 prototype 기반 cosine — radiologist-aligned metric
- **ROC-C mean** (빨간 실선): delta score — 가장 discriminative한 cosine metric
- **ROC-B/C kmeans** (점선): subtype-aware prototype 결과 비교

Val cutoff 지점은 scatter dot으로 표시됨.  
저장 파일: `triple_roc_val.png`, `triple_roc_test.png`

---

### [7] Cosine Score Distribution (`plot_cosine_distribution`)

**무엇을 보여주는가**  
HCC와 Hemangioma 각 클래스의 cosine score 분포를 KDE(Kernel Density Estimation)로 시각화한다.  
두 클래스의 분포가 얼마나 분리되어 있는지, 어느 지점이 최적 cutoff인지를 직관적으로 보여준다.

- 왼쪽 패널: HCC Cosine Score (mean prototype 기준)
- 오른쪽 패널: ΔScore (HCC − Hem cosine)
- 세로 점선: Val에서 결정된 cutoff 표시
- x축 하단: rug plot (개별 샘플 분포)

저장 파일: `cosine_dist_val.png`, `cosine_dist_test.png`

---

### [8] t-SNE (`plot_tsne`)

**무엇을 보여주는가**  
고차원 embedding space를 2D로 투영해서 클래스 분리 구조를 시각화한다.  
prototype bank가 embedding space에서 어떤 위치를 차지하는지,  
val/test 샘플이 train 분포 안에 잘 들어오는지 확인한다.

두 패널로 구성:

| 패널 | 내용 |
|---|---|
| 왼쪽 (Mean prototype) | train scatter + mean centre ★ + 1σ 타원 |
| 오른쪽 (K-Means) | train scatter + subgroup centre ★ + cluster별 1σ 타원 + cluster centre ◆ (H0, H1…) |

Val/Test 샘플은 삼각형(▲)으로 overlay됨.  
저장 파일: `tsne.png`

---

### [9] Val DeLong Test

**무엇을 보여주는가**  
두 ROC curve의 AUROC 차이가 통계적으로 유의한지 비모수 검정(DeLong, 1988)으로 확인한다.

| 비교 쌍 | 질문 |
|---|---|
| A vs B-mean | confidence AUROC와 cosine AUROC가 다른가? |
| A vs C-mean | confidence vs delta — 가장 중요한 비교 |
| A vs B-kmeans | kmeans prototype이 mean보다 나은가? |
| A vs C-kmeans | 동일 |
| B-mean vs B-kmeans | prototype 구성 방식의 차이가 유의한가? |
| C-mean vs C-kmeans | 동일 |

출력: z-statistic, p-value, ΔAUROC  
저장 파일: `delong.csv`

---

### [10] WandB Logging

**무엇을 로깅하는가**  
`wandb.run`이 활성 상태일 때 모든 수치 지표 + 그림을 자동 로깅한다.

| WandB 항목 | 내용 |
|---|---|
| `{prefix}/{model_id}/val/...` | val metrics per ROC |
| `{prefix}/{model_id}/test/...` | test metrics per ROC |
| `{prefix}/{model_id}/delong_*` | DeLong 결과 수치 |
| `{prefix}/{model_id}/triple_roc_val/test` | ROC 그림 |
| `{prefix}/{model_id}/cosine_dist_val/test` | KDE 분포 그림 |
| `{prefix}/{model_id}/tsne` | t-SNE 그림 |
| `{prefix}/{model_id}/cosine_score_table` | `wandb.Table` (샘플별 점수 전체) |
| `{prefix}/{model_id}/delong_table` | `wandb.Table` (DeLong 결과) |

---

## 추가된 기능 (`run_cosine_probe_v2`)

### [11] Reliability Diagram (`plot_reliability_diagram`)

**무엇을 보여주는가**  
"softmax output이 왜 확률이 아닌가"를 실험적으로 시각화하는 핵심 그림이다.

Confidence를 n개 구간(default 10)으로 나눈 뒤, 각 구간에서  
모델의 **평균 confidence**와 **실제 positive fraction**을 비교한다.

- 완벽히 보정된 모델: 모든 점이 y = x 대각선 위에 위치
- 실제 딥러닝 모델: 고 confidence 구간에서 실제 positive fraction보다 높게 나타나는 과확신(overconfidence)

| Calibration Error | 수식 | 의미 |
|---|---|---|
| ECE (Expected) | $\sum_b \frac{|B_b|}{N} \|{\rm acc}(B_b) - {\rm conf}(B_b)\|$ | 전체 평균 보정 오차 |
| MCE (Maximum) | $\max_b \|{\rm acc}(B_b) - {\rm conf}(B_b)\|$ | 가장 심한 구간의 보정 오차 |
| ACE (Adaptive) | 동일 샘플 수 bin 기준 | sample-imbalanced bin 보정 |

저장 파일: `reliability_val.png`, `reliability_test.png`, `calibration.csv`  
WandB: `{prefix}/{model_id}/reliability_val/test`

---

### [12] Confidence vs Cosine Scatter (`plot_conf_vs_cosine_scatter`)

**무엇을 보여주는가**  
confidence score와 cosine score가 **항상 일치하지 않는다**는 것을 보여주는 핵심 scatter plot이다.  
두 score가 다른 정보를 담고 있음을 empirically 지지한다.

x축: softmax confidence / y축: HCC Cosine Score (mean) 또는 ΔScore

cutoff 교차선으로 4개 quadrant가 나뉜다:

| Quadrant | confidence | cosine | 해석 |
|---|---|---|---|
| Q1 (↑, ↑) | HCC | HCC | 두 metric 모두 HCC — 강한 양성 |
| Q2 (↓, ↑) | Hem | HCC | **불일치** — cosine은 HCC, 모델은 Hem |
| Q3 (↓, ↓) | Hem | Hem | 두 metric 모두 Hem — 강한 음성 |
| Q4 (↑, ↓) | HCC | Hem | **불일치** — 모델은 HCC, cosine은 Hem |

Q2, Q4 영역의 케이스가 "두 score를 함께 봐야 하는 임상 상황"의 구체적 근거가 된다.

저장 파일: `conf_vs_cosine_val.png`, `conf_vs_cosine_test.png`  
WandB: `{prefix}/{model_id}/conf_vs_cosine_val/test`

---

### [13] Confusion Matrix (`plot_confusion_matrix`)

**무엇을 보여주는가**  
5개 ROC (A, B-mean, B-kmeans, C-mean, C-kmeans) × 2 split (Val, Test) = 최대 10개 confusion matrix를  
각각 normalized + raw count로 시각화한다.

각 ROC 타입이 **어떤 종류의 오류(FP vs FN)를 발생시키는지** 비교할 수 있다.  
예: cosine-based cutoff가 sensitivity를 더 높이는 대신 specificity를 낮추는 tradeoff가 있는지.

저장 파일: `cm_{split}_{roc_name}.png` (최대 10개)

---

### [14] Test-set DeLong (`run_test_delong`)

**무엇을 보여주는가**  
Val DeLong은 cutoff 결정에 사용된 같은 데이터로 검정하므로 낙관적 편향이 있다.  
**Test set에서 독립적으로 DeLong 검정**을 실행해서 발견을 재현한다.

동일한 6가지 비교 쌍을 test set labels 기준으로 실행.  
저장 파일: `delong_test.csv`

---

### [15] Per-sample WandB Image Score Table

**무엇을 보여주는가**  
각 샘플의 **이미지 + 모든 score**를 하나의 `wandb.Table`에 묶어서 로깅한다.

| 컬럼 | 내용 |
|---|---|
| `image` | 원본 입력 이미지 (최대 80 샘플) |
| `true_label` | HCC / Hemangioma |
| `confidence_score` | softmax P(HCC) |
| `hcc_cosine_mean` / `hem_cosine_mean` | mean prototype 기반 cosine |
| `delta_mean` | ΔScore (mean) |
| `hcc_cosine_kmeans` / `hem_cosine_kmeans` / `delta_kmeans` | kmeans 기반 동일 지표 |

WandB UI에서 이미지를 클릭하면서 "confidence가 높은데 cosine이 낮은 케이스"를  
직접 탐색할 수 있어 임상 오진 case study에 활용 가능하다.

WandB: `{prefix}/{model_id}/val_image_score_table`, `test_image_score_table`

---

## 출력 파일 전체 목록

```
{work_dir}/output/cosine_probe/{model_id}/
├── hcc_bank_mean.npy          # mean prototype (1, D)
├── hem_bank_mean.npy
├── hcc_bank_kmeans.npy        # kmeans prototypes (k, D)
├── hem_bank_kmeans.npy
├── hcc_assign_mean.npy        # train sample cluster assignment
├── hem_assign_mean.npy
├── hcc_assign_kmeans.npy
├── hem_assign_kmeans.npy
├── hcc_train_emb.npy          # live 추출 시 캐시
├── hem_train_emb.npy
│
├── val_scores.csv             # 샘플별 5종 score + label
├── test_scores.csv
├── cutoffs.csv                # ROC별 최적 cutoff
├── val_metrics.csv            # AUROC/Sens/Spec/PPV/NPV/F1
├── test_metrics.csv
│
├── delong.csv                 # Val DeLong 결과
├── delong_test.csv            # Test DeLong 결과 [v2 추가]
├── calibration.csv            # ECE/MCE/ACE [v2 추가]
│
├── triple_roc_val.png
├── triple_roc_test.png
├── cosine_dist_val.png
├── cosine_dist_test.png
├── tsne.png
│
├── reliability_val.png        # [v2 추가]
├── reliability_test.png
├── conf_vs_cosine_val.png     # [v2 추가]
├── conf_vs_cosine_test.png
└── cm_{split}_{roc}.png       # 최대 10개 [v2 추가]
```

---

## 사용법 요약

```python
from validation.cosine_probe import CosineProbeConfig, run_cosine_probe_v2

cfg = CosineProbeConfig(
    model_id         = model_id,
    work_dir         = cfg.work_dir,
    embedding_source = "saved",     # VICReg/ConvHybrid: "saved", SupCon/Benchmark: "live"
    n_proto          = 8,
    cutoff_strategy  = "youden",    # 또는 "sens_first"
    sens_target      = 0.90,
    tsne_perplexity  = 30.0,
    wandb_prefix     = "cosine_probe",
)

results = run_cosine_probe_v2(
    stage2_model = classifier,
    val_ds       = val_ds,
    test_ds      = test_ds,
    cfg          = cfg,
    train_ds     = train_ds,
)

# results 키:
#   banks, cutoffs, val_metrics, test_metrics,
#   delong_records, calibration_val, calibration_test, test_delong_records
```

`run_cosine_probe()` (기본) 또는 `run_cosine_probe_v2()` (전체) 중 선택.  
두 함수의 파라미터는 완전히 동일하다.
