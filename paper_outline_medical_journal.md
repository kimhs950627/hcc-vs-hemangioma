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

## 1. 서론 (Introduction)

### 1.1 HCC의 임상적 부담 (Clinical Burden of HCC)

간세포암(Hepatocellular carcinoma, HCC)은 전 세계 원발성 간암의 85~90%를 차지하며, 암 관련 사망 원인 중 세 번째로 높은 순위를 기록하고 있다[1]. 국내에서는 B형 간염바이러스(HBV) 감염이 주요 위험 인자로, 우리나라는 동아시아 지역 중에서도 HCC 발생률이 높은 나라에 속한다[2]. HCC는 조기에 발견될 경우 절제술 또는 간이식(Milan 기준 내)을 통해 5년 생존율이 70%를 초과할 수 있으나, 진행성 병기에서의 5년 생존율은 10% 미만에 불과하다[2]. 이에 대한민국 간암 진료권고안 및 AASLD, EASL, APASL 국제 가이드라인은 간경변증이나 만성 B/C형 간염 등 고위험군을 대상으로 6개월 간격의 복부 초음파(B-mode ultrasound, US) 검사를 권고하고 있다[2,3].

### 1.2 초음파 검진의 한계와 1차 의료의 현실 (Limitations of Ultrasound Surveillance and the Primary Care Gap)

복부 초음파는 비침습적이고 방사선 노출이 없으며 즉각적인 결과를 얻을 수 있는 장점이 있어 1차 의료 현장에서 폭넓게 활용된다. 그러나 조기 HCC(≤2 cm)에 대한 초음파의 단독 민감도는 약 45%에 불과하며, AFP를 병용할 경우 63%로 향상되기는 하나 여전히 제한적이다[4]. 더 근본적인 문제는 **검사자 의존성(operator dependency)**으로, 검사자에 따라 영상 품질과 판독 결과가 크게 달라져 재현성이 저하된다. 특히 전문 복부 방사선과 의사나 소화기내과 전문의가 상주하지 않는 1차 의료기관 또는 지방 병원에서는 이러한 간극이 두드러진다.

한국 1차 의료의 실제 상황을 고려할 때, AFP나 PIVKA-II 같은 혈청 종양표지자(serum tumor marker) 검사는 보험급여 기준과 의뢰 경로의 제약으로 인해 검사 자체가 어려운 경우가 많다. AFP 음성 HCC는 전체 HCC의 약 2/3를 차지하므로[5], 혈청표지자에 의존하는 선별검사만으로는 태생적 한계가 있다. 따라서 **혈청표지자나 조영제 없이 B-mode 초음파 영상만으로 HCC 가능성을 정량화할 수 있는, 검사자 독립적 방법론의 개발**은 임상적으로 매우 중요한 과제이다.

### 1.3 간 초음파 인공지능 연구의 현황과 한계 (AI in Liver Ultrasound: Prior Work and Gaps)

최근 딥러닝(deep learning, DL) 기반의 간 병변 분류 연구들은 B-mode 초음파에서 AUROC 0.83~0.94를 보고하였다[6,7]. 삼성서울병원에서도 CEUS(조영증강 초음파)를 활용하여 HCC와 FNH를 구별하는 딥러닝 모델이 연구된 바 있다[8]. 그러나 기존 연구들은 다음과 같은 방법론적 한계를 공유한다:

- 이진(binary) 분류 출력만을 제공하여, 연속형 radiologic marker로 활용할 수 없다.
- 검출 임계값(cutoff)을 테스트 데이터에서 결정함으로써 data leakage가 발생하여 보고된 성능이 과대추정될 우려가 있다.
- Decision Curve Analysis(DCA)와 같은 임상적 유용성 평가가 부재하다.
- 순수 Convolutional Neural Network(CNN) 또는 plain Vision Transformer(ViT)만을 사용하여, 소규모 의료 영상 데이터셋에서의 적합성이 제한적이다.

Plain ViT는 ImageNet 수준의 대규모 사전학습 없이는 데이터 효율이 낮아 의료 영상에서 과적합(overfitting) 위험이 높다[9]. 반면 단순 CNN은 전역적 맥락(global context) 포착에 한계가 있다. **Hybrid Vision Transformer(Hybrid ViT)**는 CNN의 귀납적 편향(locality, translation equivariance)과 Transformer의 전역 자기주의(global self-attention)를 결합하여 이러한 문제를 완화한다[10,11].

또한 기존 연구들은 모델의 의사결정 근거를 시각화하는 설명 가능성(explainability) 분석을 제공하지 않는 경우가 많다. Grad-CAM(Gradient-weighted Class Activation Mapping)[12] 또는 Transformer의 attention weight visualization을 통해 모델이 어느 영상 영역에 주목하여 HCC 점수를 산출하는지 정성적으로 확인할 수 있으며, 이는 임상 현장에서의 신뢰성 확보와 설명 가능한 AI(Explainable AI, XAI) 관점에서 중요하다.

### 1.4 연구 목적 (Study Objectives)

본 연구는 다음을 목표로 한다:

1. 삼성서울병원 단일 기관 B-mode 복부 초음파 영상 데이터셋(SMC-LUD)을 이용하여, **Hybrid Vision Transformer(CNN stem + Transformer encoder)**를 기반으로 한 분류 모델을 개발하고, 단순 cross-entropy 분류(classification-only)와 Supervised Contrastive Learning(SupCon) 정규화를 결합한 학습 방식(classification + SupCon)을 비교한다.
2. 모델 출력의 softmax 확률값인 **HCC Score = P(HCC) ∈ [0, 1]**을 연속형 radiologic marker로 정의하고, validation set만을 이용한 Youden's J 기반 최적 임계값 결정 프로토콜을 적용하여 data leakage를 원천 차단한다.
3. ROC 분석 및 Decision Curve Analysis(DCA)를 통해 HCC Score의 임상적 유용성을 정량화한다.
4. Grad-CAM 및 attention weight visualization을 통해 모델의 예측 근거를 시각화함으로써, 임상의가 이해할 수 있는 **설명 가능한 radiologic marker**임을 입증한다.
5. 1차 의료 맥락에서 혈청표지자 없이 B-mode 초음파만으로 활용 가능한 도구로서의 임상적 타당성을 제시하고, 연구의 한계와 향후 연구 방향을 명확히 제시한다.

본 연구의 novelty는 다음의 세 가지로 요약된다: (i) 연속형 HCC Score를 radiologic marker로 정의하고 검증하는 방법론적 엄밀성, (ii) Grad-CAM/attention visualization을 통한 설명 가능한 AI 적용, (iii) validation-set-only cutoff 결정으로 prospective 임상 시나리오를 재현하는 것이다.

---

## 2. 대상 및 방법 (Materials and Methods)

### 2.1 연구 설계 (Study Design)

본 연구는 단일 기관(삼성서울병원, 서울, 대한민국) 후향적 관찰 연구로, TRIPOD(Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis) 가이드라인에 따라 보고한다[13]. IRB 승인 번호: *(삽입 예정)*. 데이터 수집 기간: 2015년~2024년.

### 2.2 데이터셋: SMC-LUD (Dataset: Samsung Medical Center – Liver Ultrasound Dataset)

본 연구에서는 **SMC-LUD(Samsung Medical Center – Liver Ultrasound Dataset)**[14]를 활용하였다. SMC-LUD는 삼성서울병원에서 2015년부터 2024년까지 수집된 공개 B-mode 간 초음파 영상 데이터셋으로, Scientific Data(Nature Portfolio)에 2026년 발표된 데이터셋이다[14].

**데이터셋 주요 특성:**

| 항목 | 내용 |
|---|---|
| 출처 | 삼성서울병원(Samsung Medical Center, Seoul, Korea) |
| 수집 기간 | 2015년~2024년 |
| 전체 영상 수 | 5,385장 (환자 수: 1,021명) |
| HCC | 2,716장 (조직병리학적 확진: 수술 절제 또는 생검) |
| Hemangioma | 2,669장 (조영증강 CT/MRI 또는 추적 관찰에 의한 영상 진단) |
| 레이블링 | 전문 방사선과 의사 및 병리과 의사가 검증 |
| 분할 기준 | 환자 단위(patient-level) 분할 |
| 영상 규격 | 384 × 384 px, 흑백(grayscale), float32 |

**본 연구에 사용된 분할(split):**

| 분할 | 전체 | HCC | Hemangioma |
|---|---|---|---|
| Train | 1,858 | 972 (52.3%) | 886 (47.7%) |
| Validation | 530 | 277 (52.3%) | 253 (47.7%) |
| Test | 268 | 140 (52.2%) | 128 (47.8%) |
| **합계** | **2,656** | **1,389** | **1,267** |

모든 HCC 케이스는 조직병리학적으로 확진되었으며, 혈관종(hemangioma)은 조영증강 CT/MRI 또는 장기 추적 관찰을 통해 영상학적으로 진단되었다. 환자 단위(patient-level) 분할을 통해 동일 환자의 영상이 서로 다른 분할에 혼입되는 data leakage를 방지하였다.

**포함 기준:**
- 조직병리학적 또는 영상 기준(AASLD/EASL non-invasive criteria)으로 확진된 HCC
- 조영증강 CT/MRI 또는 추적 안정성으로 확인된 혈관종
- 기관 PACS에 B-mode 초음파 영상이 저장된 경우

**제외 기준:**
- 심한 음향 음영(acoustic shadowing) 또는 환자 움직임에 의한 영상 품질 불량
- 불명확하거나 미결(equivocal) 진단 케이스

### 2.3 모델 아키텍처: Hybrid Vision Transformer (Model Architecture)

#### 2.3.1 Encoder 설계 원칙

단순 CNN만을 사용하지 않은 이유는 다음과 같다: CNN은 지역적 텍스처(local texture) 특징 추출에 강점이 있으나, 병변 전체의 공간적 맥락(global spatial context)을 포착하는 데 한계가 있다. 반면 plain ViT는 강력한 전역 자기주의를 제공하지만 ImageNet 수준의 대규모 데이터 없이는 데이터 효율이 낮아 소규모 의료 영상에서 과적합 위험이 높다[9,11]. **Hybrid ViT**는 CNN stem이 제공하는 귀납적 편향(locality, translation equivariance)과 Transformer의 전역 모델링 능력을 결합함으로써 이러한 상충 관계를 해소한다[10]. 실제로 초음파 기반 유방암 분류 연구에서도 EfficientNetV2 + ViT의 하이브리드 구조가 단독 CNN(80%) 및 단독 ViT(89%)를 모두 상회하는 97.95% 정확도를 달성한 바 있다[11].

#### 2.3.2 CNN Backbone

본 연구에서는 CNN backbone으로 **ResNet50V2** 또는 **EfficientNetV2B0** 중 하나를 채용하였으며, 두 backbone 모두 실험 조건에 따라 평가하였다. 두 backbone 모두 ImageNet 사전학습 가중치로 초기화되었으며, 마지막 합성곱 특징 맵을 패치 토큰(patch token)으로 변환하여 Transformer encoder에 공급한다.

- **ResNet50V2**: 잔차 연결(residual connection)과 pre-activation 구조로 깊은 네트워크에서 gradient flow가 안정적이다.
- **EfficientNetV2B0**: 복합 스케일링(compound scaling)으로 파라미터 효율이 높으며, 소규모 데이터셋에서 과적합 억제에 유리하다.

#### 2.3.3 Transformer Encoder

CNN backbone의 출력 특징 맵을 flatten하여 패치 토큰 시퀀스를 구성하고, [CLS] 토큰을 앞에 붙여 Transformer encoder에 입력한다. Transformer encoder의 깊이(depth, 층 수)는 1, 2, 4, 8층 등 다양하게 설정하여 실험하였으며, 각 층은 Multi-Head Self-Attention(MHSA)과 Feed-Forward Network(FFN)으로 구성된다. 최종 [CLS] 토큰 표현을 분류 헤드의 입력으로 사용한다.

```
Input US image (384×384)
    │
    ▼
CNN Backbone (ResNet50V2 or EfficientNetV2B0)
    │  → Feature map (H' × W' × C)
    ▼
Flatten + Linear projection → Patch tokens (N, d_model)
    │
    ▼
Prepend [CLS] token
    │
    ▼
Transformer Encoder (depth = 1/2/4/8)
    │  Multi-Head Self-Attention × depth
    ▼
[CLS] token representation (B, d_model)
    │
    ▼
Classification Head: Dense(2) → Softmax
    │
    ▼
Output: {"logits": (B,2), "probabilities": (B,2)}
```

#### 2.3.4 Encoder 아키텍처 결정의 근거

본 연구에서 CNN + Transformer 하이브리드를 채택한 핵심 근거는 두 가지다. 첫째, HCC와 혈관종의 B-mode 초음파 감별에는 **텍스처(에코 패턴)와 형태(병변 경계, 내부 구조)** 모두가 중요한데, CNN은 전자에, Transformer는 후자에 각각 강점이 있다. 둘째, Transformer encoder의 attention map은 모델이 어느 영역을 근거로 예측했는지 **시각화 가능한 설명**을 제공하며, 이는 임상의와의 소통 및 XAI 관점에서 부가적 가치를 제공한다[12].

### 2.4 학습 방법론 비교 (Training Methodology Comparison)

본 연구의 핵심 비교 실험은 동일한 Hybrid ViT encoder를 사용하되, 학습 목적 함수(loss function)를 달리하는 두 조건을 비교하는 것이다. SSL(자기지도학습) 사전학습 없이 지도학습(supervised learning) 기반으로만 비교한다.

#### 2.4.1 비교 조건 설계

| 조건 | 학습 방식 | 손실 함수 |
|---|---|---|
| **Condition A** | Classification-only | Cross-entropy만 사용 |
| **Condition B** | Classification + SupCon | Cross-entropy + Supervised Contrastive Loss |

두 조건의 비교 기준은 다음과 같다:
- **정량적**: AUROC, Sensitivity, Specificity, F1-score (Test set, blind evaluation)
- **정성적**: Grad-CAM heat map 및 attention weight visualization의 임상적 해석 가능성

#### 2.4.2 Cross-Entropy (Classification-Only)

기준 모델(Condition A)은 표준적인 categorical cross-entropy loss를 사용한다:

\[
\mathcal{L}_{CE} = -\sum_{c} y_c \log \hat{p}_c
\]

여기서 \(y_c\)는 one-hot 레이블, \(\hat{p}_c\)는 softmax 출력 확률이다.

#### 2.4.3 Supervised Contrastive Learning (SupCon)

Supervised Contrastive Learning(SupCon)[15]은 동일 클래스 샘플들을 임베딩 공간에서 가깝게, 다른 클래스 샘플들을 멀리 위치시키도록 representation을 학습한다. Khosla et al.[15]은 SupCon loss가 cross-entropy에 비해 특히 **데이터 감소 환경(reduced data setting)**에서 일관된 성능 향상을 보임을 보고하였다.

SupCon loss는 다음과 같이 정의된다[15]:

\[
\mathcal{L}_{SupCon} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\mathbf{z}_i \cdot \mathbf{z}_p / \tau)}{\sum_{a \in A(i)} \exp(\mathbf{z}_i \cdot \mathbf{z}_a / \tau)}
\]

여기서 \(\mathbf{z}_i\)는 projection head를 통과한 정규화된 임베딩 벡터, \(P(i)\)는 인덱스 \(i\)와 동일 클래스인 샘플들의 집합, \(A(i)\)는 \(i\)를 제외한 전체 미니배치, \(\tau\)는 temperature hyperparameter이다.

Condition B에서는 총 손실을 다음과 같이 구성한다:

\[
\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \cdot \mathcal{L}_{SupCon}
\]

여기서 \(\lambda\)는 두 손실의 상대적 기여를 조절하는 균형 가중치(balance weight)이다. SupCon loss는 분류 헤드와 별도의 projection head를 통과한 [CLS] 표현에 적용되며, 미니배치 내 동일 클래스 쌍(positive pair)을 자동으로 구성하여 contrastive learning을 수행한다.

#### 2.4.4 학습 설정 (Training Configuration)

- **입력 크기**: 384 × 384 px, grayscale, float32 정규화
- **배치 크기**: GPU 메모리(12 GPU-hour 제약) 내 최대화
- **옵티마이저**: Adam (lr = 1e-4, weight decay 적용)
- **학습률 스케줄**: Cosine annealing with warmup
- **데이터 증강**:
  - 무작위 수평/수직 반전
  - 무작위 밝기/대비 조정 (achromatic channel만; US 영상 특성 고려)
  - Gaussian blur (σ ∈ [0.1, 2.0])
  - 무작위 crop & resize (→ 384 × 384)
- **학습 환경**: Kaggle GPU (P100/T4), 12 GPU-hour 제약 이내

### 2.5 HCC Score 정의 (HCC Score Definition)

> **HCC Score = P(HCC) = softmax 출력의 인덱스 1에 해당하는 확률값 ∈ [0, 1]**

HCC Score는 단순한 이진 분류기의 출력이 아닌, **AFP나 PIVKA-II와 같은 혈청 바이오마커에 유사한 연속형 radiologic marker**로 정의한다. 점수가 1에 가까울수록 HCC 가능성이 높고, 0에 가까울수록 혈관종(양성 병변) 가능성이 높다. 이 연속형 출력은 단일 임계값(binary cutoff) 적용 이전에도 그 자체로 임상적 정보를 담고 있으며, 임계값 선택의 유연성을 보장한다.

### 2.6 임계값 결정 방법론 (Cutoff Determination: No Data Leakage)

최적 임계값은 **validation set만을 사용하여** 결정하며, test set은 단 한 번의 맹검 평가(blind evaluation)에만 사용한다. 이는 test data에서 임계값을 결정할 경우 민감도/특이도가 과대추정되어 임상적 의미가 없어지기 때문이다.

| 전략 | 정의 | 역할 |
|---|---|---|
| **Youden's J** (1차) | argmax(Sensitivity + Specificity − 1) on Val ROC | 주 보고 임계값 |
| Sensitivity-first (2차) | Val ROC에서 Sensitivity ≥ 0.90을 만족하는 최고 임계값 | 민감도 우선 분석 |

Youden's J 전략은 민감도와 특이도의 균형 있는 최적화를 위한 표준적 방법이며, 간암 선별검사와 같이 민감도와 특이도 모두 중요한 임상 상황에 적합하다.

### 2.7 시각화: Grad-CAM 및 Attention Weight (Visualization: Grad-CAM and Attention Weights)

모델의 예측 근거를 시각적으로 확인하기 위해 두 가지 설명 가능성 기법을 적용한다:

- **Grad-CAM(Gradient-weighted Class Activation Mapping)**[12]: CNN backbone의 마지막 합성곱 층에 대한 역전파 기울기를 이용하여, HCC 예측에 기여한 영상 영역을 heat map으로 시각화한다. 수식적으로는 다음과 같다[12]:

\[
L^c_{Grad\text{-}CAM} = \text{ReLU}\!\left(\sum_k \alpha_k^c A^k\right), \quad \alpha_k^c = \frac{1}{Z}\sum_i\sum_j \frac{\partial y^c}{\partial A^k_{ij}}
\]

여기서 \(A^k\)는 \(k\)번째 feature map, \(y^c\)는 클래스 \(c\)의 score (softmax 이전), \(\alpha_k^c\)는 전역 평균 풀링(global average pooling)된 기울기이다.

- **Attention Weight Visualization**: Transformer encoder의 [CLS] 토큰과 패치 토큰 간의 cross-attention(CA) 또는 self-attention 가중치를 추출하여, 모델이 주목한 공간적 위치를 원본 영상 위에 overlay한다.

이러한 시각화는 임상의가 모델의 판단 근거를 직관적으로 확인하고, 오분류(misclassification) 사례를 분석하는 데 활용될 수 있다.

### 2.8 통계 분석 (Statistical Analysis)

- **AUROC**: DeLong 방법을 이용한 95% 신뢰구간 산출[16]
- **임계값 적용 시 지표**: 민감도(Sensitivity), 특이도(Specificity), 양성예측도(PPV), 음성예측도(NPV), F1-score, 혼동 행렬(TP/FP/TN/FN)
- **Decision Curve Analysis(DCA)**: 임계 확률 0.05~0.95 범위에서 treat-all 및 treat-none 대비 net benefit 비교[17]
- **Bootstrap 재추출** (n = 1,000회): Test set 전 지표에 대한 95% CI 산출
- **Score 분포 시각화**: Train/Val/Test 각각에 대한 클래스별 Gaussian KDE 분포도
- **소프트웨어**: Python 3.11, TensorFlow/Keras 3, scikit-learn, scipy, matplotlib

---

### 한계 (Limitations)

본 연구는 다음과 같은 한계를 가진다:

1. **단일 기관 후향적 연구(Single-center retrospective design)**: 삼성서울병원 단일 기관에서 수집된 데이터로, 다른 기관의 초음파 장비, 검사자 스타일, 환자군에 대한 외적 타당도(external validity)는 확인되지 않았다. 다기관 전향적 연구를 통한 검증이 필요하다.

2. **AFP 등 혈청표지자와의 비교 부재**: 본 데이터셋에는 AFP, PIVKA-II 등 기존 혈청 종양표지자 데이터가 포함되어 있지 않아, 기존 표지자 대비 HCC Score의 incremental benefit을 직접 비교하는 것이 불가능하였다. 특히 AFP 음성 HCC 아군에서의 성능을 별도 분석할 수 없었다는 점은 중요한 한계이다.

3. **GPU 및 연산 자원의 제약**: 본 연구의 모든 실험은 Kaggle GPU 환경에서 12 GPU-hour 이내의 제약 조건 하에 수행되었다. 이로 인해 더 깊은 Transformer encoder 설정, 대규모 배치, 또는 더 많은 에폭(epoch) 수의 학습은 시도하지 못하였다. 충분한 연산 자원이 확보된다면 추가적인 아키텍처 탐색이 가능할 것이다.

4. **이진 대조군의 한계**: 본 연구의 대조군은 혈관종만으로 구성되어 있다. 실제 임상에서 간 병변의 감별 진단에는 국소결절성과증식(FNH), 재생 결절(regenerative nodule), 전이성 간암(metastasis) 등이 포함되며, 본 연구의 이진 분류 설계는 이러한 다양한 병변을 반영하지 못한다.

5. **방사선과 의사와의 직접 성능 비교 부재**: 동일 데이터셋에서 전문 방사선과 의사의 판독 성능과 본 모델의 성능을 직접 비교하지 못하였다.

6. **병변 크기 층화 분석 불가**: 조기 HCC(≤2 cm)와 진행성 HCC를 구분하는 층화 분석이 현재 데이터셋 구조에서는 시행되지 않았다.

이러한 한계를 극복하기 위해, 향후 연구에서는 (i) 다기관 전향적 코호트에서의 외적 타당도 검증, (ii) AFP/PIVKA-II를 포함한 혈청표지자와의 병용 분석, (iii) 병변 크기 및 기저 간질환 중증도에 따른 층화 분석, (iv) 방사선과 의사와의 head-to-head 비교 연구가 필요하다.

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

## 6. 참고문헌 (References)

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. *CA Cancer J Clin.* 2021;71(3):209-249.

2. Korean Liver Cancer Association; National Cancer Center. 2022 KLCA-NCC Korea Practice Guidelines for the Management of Hepatocellular Carcinoma. *Korean J Radiol.* 2022;23(12):1126-1240.

3. European Association for the Study of the Liver. EASL Clinical Practice Guidelines: Management of hepatocellular carcinoma. *J Hepatol.* 2018;69(1):182-236.

4. Tzartzeva K, Obi J, Rich NE, et al. Surveillance Imaging and Alpha Fetoprotein for Early Detection of Hepatocellular Carcinoma in Patients With Cirrhosis: A Meta-analysis. *Gastroenterology.* 2018;154(6):1706-1718.e1.

5. Tsuchiya N, Sawada Y, Endo I, et al. Biomarkers for the early diagnosis of hepatocellular carcinoma. *World J Gastroenterol.* 2015;21(37):10573-10583.

6. Yang Q, Wei J, Hao X, et al. Improving B-mode ultrasound diagnostic performance for focal liver lesions using deep learning: A multicentre study. *EBioMedicine.* 2020;56:102777.

7. Zhang WB, Chen YN, Zeng MS, et al. Deep learning to diagnose AFP-negative hepatocellular carcinoma from focal nodular hyperplasia on contrast-enhanced ultrasound. *Front Oncol.* 2022;12:843763.

8. Kim JH, Kim SY, Kim ER, et al. Deep learning classification of focal liver lesions with contrast-enhanced ultrasound from arterial phase recordings. *Presented at Seoul National University Hospital; Samsung Medical Center collaborative study.* 2023. (Available at: https://snu.elsevierpure.com/en/publications/deep-learning-classification-of-focal-liver-lesions-with-contrast)

9. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In: *Proceedings of International Conference on Learning Representations (ICLR).* 2021.

10. Chen CF, Fan Q, Panda R. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).* 2021:357-366.

11. Hamuod AK, Abdelhamid AA, Ibrahim A, et al. A Hybrid Model for Ultrasound Image-Based Breast Cancer Diagnosis Using EfficientNet-V2 and Vision Transformer. *Diagnostics (Basel).* 2026;16(8):1176.

12. Selvaraju RR, Cogswell M, Das A, Vedantam R, Parikh D, Batra D. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV).* 2017:618-626.

13. Collins GS, Reitsma JB, Altman DG, Moons KG. Transparent reporting of a multivariable prediction model for individual prognosis or diagnosis (TRIPOD): the TRIPOD statement. *BMJ.* 2015;350:g7594.

14. Shin SH, Kim TH, Park J, et al. SMC-LUD: Large-Scale B-Mode Liver Ultrasound Dataset for Hepatocellular Carcinoma and Hemangioma Classification. *Sci Data.* 2026;13(1):649.

15. Khosla P, Tian Y, Wang C, et al. Supervised Contrastive Learning. In: *Advances in Neural Information Processing Systems (NeurIPS).* 2020;33:18661-18673.

16. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics.* 1988;44(3):837-845.

17. Vickers AJ, Elkin EB. Decision curve analysis: a novel method for evaluating prediction models. *Med Decis Making.* 2006;26(6):565-574.

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

*Last updated: 2026-06-25 (Section 1 & 2 revised with SupCon comparison, SMC-LUD details, Grad-CAM XAI, Limitations)*  
*Author: Kim Hyun-Soo, M.D. — Department of Family Medicine, Jeonju, Republic of Korea*
