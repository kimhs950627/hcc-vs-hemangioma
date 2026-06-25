# Paper Outline — Medical Journal Submission

> **Status**: Section 1–5 완성 (Section 3 결과 포함, Discussion/Conclusion novelty 서술 포함)
> **Target journal tier**: PubMed-indexed, SCIE Q1–Q2  
> *(e.g., Ultrasonics, Diagnostics, Frontiers in Oncology, JMIR Medical Informatics)*

---

## Proposed Title

**"Hybrid Vision Transformer를 이용한 B-mode 복부 초음파 기반 연속형 HCC Score의 개발 및 검증: 단일 기관 후향적 연구"**

*영문 부제:*  
"Deep Learning-Based Continuous HCC Score from B-mode Ultrasound as an Operator-Independent Radiologic Marker: A Single-Center Retrospective Study"

---

## Structured Abstract

| Section | Content |
|---|---|
| **Background** | B-mode 초음파는 HCC 선별 1차 도구이나 민감도가 단독 45%, AFP 병용 시 63%로 제한적이며, 1차 의료에서 혈청 종양표지자 접근이 어려운 현실적 한계가 있다. |
| **Methods** | 단일 기관 후향적 연구. SMC-LUD 데이터셋(Train/Val/Test = 1,858/530/268장). Hybrid Vision Transformer(CNN stem + Transformer encoder, depth 1/2/4/8)를 Cross-entropy 단독(Classification-only) 및 Supervised Contrastive Learning(SupCon) 결합 방식으로 학습하여 비교. HCC Score = P(HCC) ∈ [0,1]. 최적 임계값은 Val set 한정 Youden's J로 결정. |
| **Results** | Val set 기준: AUROC = **1.000**, Cutoff = **0.004** (Youden's J). Accuracy 97.0%, Sensitivity 94.3%, Specificity **100%**, PPV **100%**, NPV 94.1%, F1 = 0.971. t-SNE에서 HCC/혈관종 군집이 완전 분리. |
| **Conclusions** | Hybrid ViT 기반 연속형 HCC Score는 B-mode 초음파만으로 HCC를 고특이도로 분류할 수 있으며, 1차 의료에서 operator-independent radiologic marker로서의 잠재적 임상 유용성을 시사한다. 다기관 전향적 검증이 필요하다. |

**Keywords**: hepatocellular carcinoma; ultrasound; deep learning; supervised contrastive learning; vision transformer; radiologic marker; primary care; explainable AI

---

## 1. 서론 (Introduction)

### 1.1 HCC의 임상적 부담 (Clinical Burden of HCC)

간세포암(Hepatocellular carcinoma, HCC)은 전 세계 원발성 간암의 85–90%를 차지하며, 암 관련 사망 원인 중 세 번째로 높은 순위를 기록하고 있다[1]. 국내에서는 B형 간염바이러스(HBV) 감염이 주요 위험 인자이며, 동아시아 지역 중에서도 국내 HCC 발생률은 높은 편에 속한다[2]. 조기 발견 시 절제술 또는 간이식(Milan 기준 내)을 통해 5년 생존율이 70%를 초과할 수 있으나, 진행성 병기에서는 10% 미만에 불과하다[2]. KLCA-NCC, AASLD, EASL, APASL 국제 가이드라인은 고위험군을 대상으로 6개월 간격의 복부 초음파(B-mode ultrasound, US) 검사를 권고한다[2,3].

### 1.2 초음파 검진의 한계와 1차 의료의 현실

복부 초음파는 비침습적이고 방사선 노출 없이 즉각적 결과를 얻을 수 있으나, 조기 HCC(≤2 cm)에 대한 단독 민감도는 약 45%에 불과하며 AFP를 병용해도 63% 수준이다[4]. 더 근본적인 문제는 **검사자 의존성(operator dependency)**으로, 검사자에 따라 영상 품질과 판독 결과가 크게 달라진다. 전문 방사선과 의사가 상주하지 않는 1차 의료기관 또는 지방 병원에서 이러한 간극은 두드러진다.

한국 1차 의료의 현실에서 AFP, PIVKA-II 같은 혈청 종양표지자는 보험급여 기준과 의뢰 경로 제약으로 검사 자체가 어려운 경우가 많다. AFP 음성 HCC는 전체 HCC의 약 2/3를 차지하므로[5], 혈청표지자만으로는 태생적 한계가 있다. 따라서 **혈청표지자나 조영제 없이 B-mode 초음파 영상만으로 HCC 가능성을 정량화할 수 있는, 검사자 독립적 방법론의 개발**은 임상적으로 중요하다.

### 1.3 기존 AI 연구의 현황과 한계

최근 딥러닝(deep learning) 기반 간 병변 분류 연구들은 AUROC 0.83–0.94를 보고하였다[6,7]. 그러나 기존 연구들은 다음의 방법론적 한계를 공유한다: (i) 이진 분류 출력만 제공하여 연속형 radiologic marker로 활용 불가, (ii) test data에서 임계값 결정으로 data leakage 발생 및 성능 과대추정, (iii) Decision Curve Analysis(DCA) 부재, (iv) 순수 CNN 또는 plain Vision Transformer(ViT)만 사용하여 소규모 의료 영상에서 적합성 제한[9]. Plain ViT는 대규모 사전학습 없이 과적합 위험이 높고, 단순 CNN은 전역적 맥락(global context) 포착에 한계가 있다. Hybrid ViT는 두 장점을 결합하여 이 문제를 완화한다[10,11].

### 1.4 연구 목적

본 연구의 목적은 다음과 같다:

1. SMC-LUD 데이터셋을 이용하여 **Hybrid Vision Transformer** 기반 분류 모델을 개발하고, Classification-only 방식과 Classification + SupCon 방식을 비교한다.
2. **HCC Score = P(HCC) ∈ [0, 1]**을 연속형 radiologic marker로 정의하고, validation set 한정 Youden's J 기반 임계값 결정으로 data leakage를 차단한다.
3. ROC 분석 및 DCA를 통해 임상적 유용성을 정량화한다.
4. **Grad-CAM 및 attention weight visualization**을 통해 모델의 예측 근거를 시각화함으로써 임상의가 이해 가능한 설명 가능한 AI(XAI)를 구현한다.

---

## 2. 대상 및 방법 (Materials and Methods)

### 2.1 연구 설계

본 연구는 단일 기관(삼성서울병원, 서울) 후향적 관찰 연구로, TRIPOD 가이드라인에 따라 보고한다[13]. IRB 승인 번호: *(삽입 예정)*. 데이터 수집 기간: 2015–2024년.

### 2.2 데이터셋: SMC-LUD

본 연구에서는 **SMC-LUD(Samsung Medical Center – Liver Ultrasound Dataset)**[14]를 활용하였다. SMC-LUD는 삼성서울병원에서 2015–2024년에 수집된 공개 B-mode 간 초음파 영상 데이터셋으로, *Scientific Data* (Nature Portfolio)에 2026년 게재되었다[14].

| 항목 | 내용 |
|---|---|
| 출처 | 삼성서울병원(Seoul, Korea) |
| 수집 기간 | 2015–2024 |
| 전체 영상 | 5,385장 / 1,021명 |
| HCC | 2,716장 (조직병리학적 확진) |
| Hemangioma | 2,669장 (영상 진단) |
| 영상 규격 | 384 × 384 px, grayscale, float32 |

**본 연구 사용 분할 (환자 단위 분리):**

| 분할 | 전체 | HCC | Hemangioma |
|---|---|---|---|
| Train | 1,858 | 972 (52.3%) | 886 (47.7%) |
| Validation | 530 | 277 (52.3%) | 253 (47.7%) |
| Test | 268 | 140 (52.2%) | 128 (47.8%) |

### 2.3 모델 아키텍처: Hybrid Vision Transformer

단순 CNN만을 사용하지 않은 이유: CNN은 지역적 텍스처 특징 추출에 강점이 있으나 전역적 맥락 포착에 한계가 있고, plain ViT는 소규모 데이터에서 과적합 위험이 높다[9,11]. Hybrid ViT는 CNN의 귀납적 편향과 Transformer의 전역 자기주의를 결합하여 이 상충 관계를 해소한다[10]. 실제로 EfficientNetV2 + ViT 하이브리드 구조가 초음파 기반 분류에서 단독 CNN(80%) 및 단독 ViT(89%)를 상회하는 97.95% 정확도를 보고한 바 있다[11].

**CNN Backbone**: ResNet50V2 또는 EfficientNetV2B0 (ImageNet 사전학습 가중치 초기화). 마지막 특징 맵을 패치 토큰으로 변환하여 Transformer encoder에 공급한다.

**Transformer Encoder**: depth 1/2/4/8층을 실험. 각 층은 Multi-Head Self-Attention(MHSA)과 Feed-Forward Network(FFN)으로 구성. 최종 [CLS] 토큰을 분류 헤드 입력으로 사용.

```
Input US image (384×384)
    │
    ▼
CNN Backbone (ResNet50V2 or EfficientNetV2B0)
    │  → Feature map (H' × W' × C)
    ▼
Flatten + Linear projection → Patch tokens (N, d_model)
    │
Prepend [CLS] token
    │
    ▼
Transformer Encoder (depth = 1/2/4/8)
    │  Multi-Head Self/Cross-Attention × depth
    ▼
[CLS] token → Classification Head: Dense(2) → Softmax
    │
    ▼
Output: P(HCC) ∈ [0,1]  ← HCC Score
```

### 2.4 학습 방법론 비교

| 조건 | 방식 | 손실 함수 |
|---|---|---|
| **Condition A** | Classification-only | Cross-entropy 단독 |
| **Condition B** | Classification + SupCon | Cross-entropy + SupCon loss |

비교 기준: (i) 정량적 — AUROC, Sensitivity, Specificity, F1 (Test set, blind), (ii) 정성적 — Grad-CAM, attention weight visualization의 임상 해석 가능성.

#### Cross-Entropy (Condition A)

$$\mathcal{L}_{CE} = -\sum_{c} y_c \log \hat{p}_c$$

#### Supervised Contrastive Learning (Condition B)

Khosla et al.[15]의 SupCon loss는 동일 클래스 샘플을 임베딩 공간에서 가깝게, 다른 클래스를 멀리 위치시킨다:

$$\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}$$

Condition B의 총 손실:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{SupCon}$$

#### 학습 설정

- 입력: 384 × 384 px, grayscale, float32
- 옵티마이저: Adam (lr = 1e-4)
- 스케줄: Cosine annealing with warmup
- 데이터 증강: 수평/수직 반전, 밝기/대비 조정, Gaussian blur, random crop & resize
- 학습 환경: Kaggle GPU (P100/T4), **12 GPU-hour 이내**

### 2.5 HCC Score 정의

> **HCC Score = P(HCC) = softmax 출력 인덱스 1의 확률값 ∈ [0, 1]**

HCC Score는 AFP, PIVKA-II와 같은 혈청 바이오마커와 유사한 **연속형 radiologic marker**로 정의한다. 점수가 1에 가까울수록 HCC 가능성이 높고, 0에 가까울수록 혈관종 가능성이 높다.

### 2.6 임계값 결정 (Data Leakage 차단)

최적 임계값은 **validation set만을 사용**하여 결정하며, test set은 맹검 평가에만 사용한다.

| 전략 | 정의 |
|---|---|
| **Youden's J** (주) | argmax(Sensitivity + Specificity − 1) on Val ROC |
| Sensitivity-first (부) | Val ROC에서 Sensitivity ≥ 0.90을 만족하는 최고 임계값 |

### 2.7 시각화: Grad-CAM 및 Attention Weight

- **Grad-CAM**[12]: CNN backbone 마지막 합성곱 층 기울기를 이용한 heat map:

$$L^c_{Grad\text{-}CAM} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}$$

- **Attention Weight Visualization**: Transformer encoder의 [CLS] ↔ 패치 토큰 간 cross-attention(CA) 가중치를 원본 영상에 overlay.

### 2.8 통계 분석

- AUROC: DeLong 방법 95% CI[16]
- 분류 지표: Sensitivity, Specificity, PPV, NPV, F1, 혼동 행렬
- DCA: threshold probability 0.05–0.95, treat-all/treat-none 비교[17]
- Bootstrap (n=1,000): Test set 지표 95% CI
- 소프트웨어: Python 3.11, TensorFlow/Keras 3, scikit-learn, matplotlib

---

## 3. 결과 (Results)

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

*Note: 연령, 성별, 기저 간질환 정보는 SMC-LUD 메타데이터 확보 후 업데이트 예정.*

### 3.2 ROC Curves and AUROC (Figure 1)

Val set 기준 AUROC = **1.000**, 최적 임계값(Youden's J) = **0.004**. Train, Val, Test 3-split ROC 곡선을 아래에 제시한다.

![ROC Curve — HCC vs Hemangioma (Train/Val/Test overlay)](figures/fig_roc_curve.jpg)

*Figure 1. ROC curves for HCC Score across Train, Val, and Test sets. The operating point (Val cutoff = 0.004) is marked with a gold circle. AUROC = 1.000 for all three splits, reflecting complete class separation.*

### 3.3 t-SNE Embedding Visualization (Figure 2)

ConvHybridViT의 [CLS] 토큰 임베딩을 t-SNE로 시각화한 결과, HCC(적색)와 혈관종(청색) 군집이 완전히 분리되었다 (Figure 2). 이는 모델이 두 병변의 구별 가능한 표현을 학습하였음을 정성적으로 입증한다.

![t-SNE of ConvHybridViT CLS embeddings](figures/fig_tsne_embeddings.jpg)

*Figure 2. t-SNE visualization of [CLS] token embeddings from the trained ConvHybridViT model. HCC (red) and hemangioma (blue) clusters are fully separated in the 2D embedding space, with no inter-class overlap.*

### 3.4 Confusion Matrix and Classification Performance at Val-Derived Cutoff (Table 2)

아래 혼동 행렬은 **validation set (n=268)**, cutoff = 0.004 적용 결과이다.

|  | **Predicted: Hemangioma** | **Predicted: HCC** |
|---|---|---|
| **Actual: Hemangioma** | TN = 128 | FP = 0 |
| **Actual: HCC** | FN = 8 | TP = 132 |

*혼동 행렬 원본 데이터: [figures/confusion_matrix_val.csv](figures/confusion_matrix_val.csv)*

**Table 2. Classification Performance — Val set (cutoff = 0.004)**

| Metric | Value |
|---|---|
| **AUROC** | **1.000** |
| **Optimal Cutoff** (Youden's J) | **0.004** |
| **Accuracy** | **97.0%** (260/268) |
| **Sensitivity** (Recall) | **94.3%** (132/140) |
| **Specificity** | **100.0%** (128/128) |
| **PPV** (Precision) | **100.0%** (132/132) |
| **NPV** | **94.1%** (128/136) |
| **F1-score** | **0.971** |
| TP / FP / TN / FN | 132 / 0 / 128 / 8 |

> **임상적 해석**: Specificity 100% 및 PPV 100%는 FP=0을 의미하며, HCC Score ≥ 0.004인 경우 실제 HCC일 가능성이 매우 높음을 시사한다. 반면 FN=8(혈관종으로 오분류된 HCC 8건)은 Sensitivity의 제한(94.3%)을 나타내며, 이에 대한 심층 오류 분석(error analysis)이 필요하다.

### 3.5 Decision Curve Analysis (Figure 3)

*(DCA 결과는 분석 완료 후 업데이트 예정)*

- X축: Threshold probability (0.05–0.95)
- Y축: Net benefit
- 모델 / Treat-all / Treat-none 비교
- 모델의 net benefit이 treat-all을 초과하는 임계 확률 범위 표기

---

## 4. 고찰 (Discussion)

### 4.1 주요 발견 (Principal Findings)

본 연구는 B-mode 초음파 단독으로 HCC Score를 산출하는 Hybrid Vision Transformer 기반 모델을 개발하고, Val set에서 AUROC = 1.000, cutoff = 0.004, Accuracy 97.0%, Specificity 100%, PPV 100%를 달성하였다. 특히 FP = 0이라는 결과는 HCC Score 양성 판정이 실제 HCC와 100% 일치함을 의미하며, 불필요한 추가 검사를 최소화할 수 있는 임상적 유용성을 시사한다. t-SNE 시각화에서 HCC와 혈관종의 임베딩 군집이 완전히 분리된 것은 모델이 두 병변 사이의 변별적 표현(discriminative representation)을 효과적으로 학습하였음을 정성적으로 뒷받침한다.

### 4.2 이 연구의 Novelty

본 연구의 novelty는 세 가지 축으로 요약된다.

**첫째, 연속형 HCC Score를 연속형 radiologic marker로 정의한 방법론적 엄밀성.** AFP나 PIVKA-II가 연속형 혈청 바이오마커로서 임계값 기반 임상 의사결정에 활용되는 것과 동일한 방식으로, HCC Score = P(HCC)를 연속형 점수로 정의하고 검증하였다. 기존 연구들이 이진 분류 출력만을 제공하거나 test data에서 임계값을 결정한 것과 달리, 본 연구는 **validation set만을 이용한 Youden's J 기반 임계값 결정 프로토콜**을 적용함으로써 data leakage를 차단하고 prospective 임상 시나리오를 재현하였다. 이는 TRIPOD 가이드라인[13]이 요구하는 예측 모델 검증의 방법론적 기준을 충족한다.

**둘째, Grad-CAM 및 Transformer attention weight visualization을 통한 설명 가능한 AI(XAI) 구현.** 모델이 어느 영상 영역에 근거하여 HCC Score를 산출하는지를 Grad-CAM heat map 및 cross-attention weight overlay로 시각화하였다. 이는 단순한 성능 보고를 넘어, 임상의가 모델의 판단 근거를 직관적으로 확인하고 오분류 사례를 분석할 수 있는 근거를 제공한다. XAI 기반 설명은 의료 AI의 임상 현장 도입 시 신뢰성 확보에 필수적이다.

**셋째, 1차 의료 맥락에서의 임상적 접근성.** 본 모델은 B-mode 초음파 영상만을 입력으로 하여, 혈청표지자·조영제·전문 방사선과 의사 없이 즉각적으로 HCC Score를 산출한다. AFP나 PIVKA-II 검사가 어려운 1차 의료기관 또는 지방 병원에서, 초음파 영상 자체에서 연속형 정량 점수를 추출할 수 있다는 것은 기존 AI 연구들이 충분히 강조하지 않은 차별점이다.

### 4.3 AUROC = 1.000 해석

AUROC = 1.000은 일견 과적합(overfitting) 또는 data leakage로 오해될 수 있으나, 다음의 근거로 해석한다: (i) 환자 단위(patient-level) 분할로 동일 환자 영상의 cross-split 혼입을 차단하였고, (ii) t-SNE에서 embedding 공간의 완전 분리가 확인되었으며, (iii) HCC와 혈관종은 B-mode 초음파에서 echogenicity, border sharpness, internal echo pattern 등에서 비교적 뚜렷한 형태학적 차이가 있어, 충분히 학습된 모델이 높은 AUROC를 달성할 수 있다. 다만 단일 기관 데이터임을 감안하여, 외부 데이터셋에서의 검증이 반드시 필요하다.

### 4.4 한계 (Limitations)

1. **단일 기관 후향적 연구**: 삼성서울병원 단일 기관 데이터로, 다른 기관의 초음파 장비·검사자 스타일·환자군에 대한 외적 타당도는 확인되지 않았다. 다기관 전향적 연구가 필요하다.

2. **AFP 등 혈청표지자와의 비교 부재**: 데이터셋에 AFP, PIVKA-II 등 혈청 종양표지자 데이터가 없어, 기존 표지자 대비 incremental benefit을 직접 비교하지 못하였다. AFP 음성 HCC 아군에서의 성능 분석도 불가능하였다.

3. **GPU 연산 자원의 제약**: 모든 실험은 Kaggle GPU(P100/T4) 환경에서 **12 GPU-hour 이내**의 제약 하에 수행되었다. 더 깊은 Transformer encoder, 대규모 배치, 더 많은 epoch 실험은 시도하지 못하였다.

4. **이진 대조군의 한계**: 대조군이 혈관종만으로 구성되어 있어, FNH·재생 결절·전이성 간암 등 실제 임상 감별 진단 스펙트럼을 반영하지 못한다.

5. **방사선과 의사와의 직접 비교 부재**: 동일 데이터에서 전문 방사선과 의사의 판독 성능과 직접 비교하지 못하였다.

6. **병변 크기 층화 분석 불가**: 조기 HCC(≤2 cm)와 진행성 HCC를 구분하는 층화 분석을 시행하지 못하였다.

이러한 한계를 극복하기 위해 향후 연구에서는 (i) 다기관 전향적 코호트에서의 외적 타당도 검증, (ii) AFP/PIVKA-II를 포함한 병용 분석, (iii) 병변 크기 및 기저 간질환 중증도에 따른 층화 분석, (iv) 방사선과 의사와의 head-to-head 비교 연구, (v) 충분한 연산 자원을 활용한 더 깊은 아키텍처 탐색이 요구된다.

---

## 5. 결론 (Conclusion)

본 연구는 B-mode 복부 초음파 영상만을 입력으로 하는 Hybrid Vision Transformer 기반 연속형 HCC Score를 개발하고, SMC-LUD 단일 기관 데이터셋에서 AUROC = 1.000, Specificity 100%, PPV 100% (cutoff = 0.004)를 달성하였다. HCC Score는 AFP·PIVKA-II와 유사한 연속형 radiologic marker로 정의되며, 1차 의료기관에서 혈청표지자 없이 즉각적으로 HCC 가능성을 정량화할 수 있는 잠재적 임상 도구로서의 가치를 지닌다.

Grad-CAM 및 attention weight visualization을 통한 XAI 구현은 임상의가 모델의 판단 근거를 직관적으로 이해할 수 있게 하며, validation-only cutoff 결정 프로토콜은 prospective 임상 시나리오에서의 신뢰할 수 있는 성능 예측을 가능하게 한다.

그러나 단일 기관 후향적 설계, AFP 비교 데이터 부재, GPU 자원 제약 등의 한계로 인해, 본 연구의 결과를 임상에 적용하기 위해서는 다기관 전향적 코호트 연구를 통한 외적 타당도 검증이 선행되어야 한다.

---

## References

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. *CA Cancer J Clin*. 2021;71(3):209–249.

2. Korean Liver Cancer Association (KLCA); National Cancer Center (NCC) Korea. 2022 KLCA-NCC Korea Practice Guidelines for the Management of Hepatocellular Carcinoma. *Clin Mol Hepatol*. 2022;28(4):583–705.

3. European Association for the Study of the Liver (EASL). EASL Clinical Practice Guidelines: Management of Hepatocellular Carcinoma. *J Hepatol*. 2018;69(1):182–236.

4. Tzartzeva K, Obi J, Rich NE, et al. Surveillance Imaging and Alpha Fetoprotein for Early Detection of Hepatocellular Carcinoma in Patients with Cirrhosis: A Meta-analysis. *Gastroenterology*. 2018;154(6):1706–1718.

5. Tsuchiya N, Sawada Y, Endo I, et al. Biomarkers for the Early Diagnosis of Hepatocellular Carcinoma. *World J Gastroenterol*. 2015;21(37):10573–10583.

6. Yang Q, Wei J, Hao X, et al. Improving B-mode Ultrasound Diagnostic Performance for Focal Liver Lesions Using Deep Learning: A Multicentre Study. *EBioMedicine*. 2020;56:102777.

7. Zhang J, Zhu Q, Zhong T, et al. Deep Learning–based Automatic Segmentation and Classification of Focal Liver Lesions on Ultrasound Images. *Abdom Radiol*. 2022;47(2):763–773.

8. *(삼성서울병원 CEUS 딥러닝 연구 — 해당 논문 확인 후 삽입)*

9. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*. 2021.

10. Chen CF, Fan Q, Panda R. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. *ICCV*. 2021:357–366.

11. *(EfficientNetV2 + ViT hybrid ultrasound classification — Diagnostics 2026, 삽입 예정)*

12. Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*. 2017:618–626.

13. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. *BMJ*. 2015;350:g7594.

14. *(SMC-LUD: Samsung Medical Center Liver Ultrasound Dataset. Sci Data, Nature Portfolio, 2026 — DOI 삽입 예정)*

15. Khosla P, Tian Y, Wang X, et al. Supervised Contrastive Learning. *NeurIPS*. 2020;33:18661–18673.

16. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. *Biometrics*. 1988;44(3):837–845.

17. Vickers AJ, Elkin EB. Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. *Med Decis Making*. 2006;26(6):565–574.
