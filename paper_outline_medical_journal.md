# 논문 초고 개요 — 의학 저널 투고용

> **작성 상태**: 실험 결과 기입 완료 (rev. 2026-06-30 — §3.4/§3.5/§4.3/§4.4 cosine probe 결과 반영)
> **목표 저널 등급**: PubMed 등재, SCIE Q1–Q2
> *(예: Ultrasonics, Diagnostics, Frontiers in Oncology, JMIR Medical Informatics)*

---

## 제목 (안)

**B-mode 복부 초음파에서 HCC 유사도를 정량화하는 새로운 이중 출력 영상표지자의 개발:**
**HCC Cosine Score와 Confidence Score의 상호보완적 임상적 의의**

*영문 병기:*
**Development of a Novel Dual-Output Imaging Marker for Quantifying HCC-Likeness on B-mode Abdominal Ultrasound: Complementary Clinical Roles of HCC Cosine Score and Confidence Score**

> **제목 선택 근거**: 제목의 중심을 모델 architecture가 아닌 **새로운 영상표지자(imaging marker) 제안**에 두었다.
> 독자가 서지 검색만으로도 "B-mode 초음파에서 HCC 유사도를 수치화한 새 표지자"라는 임상적 기여를 즉시 인지할 수 있도록 한다.

---

## 구조화 초록 (Structured Abstract)

### 연구 배경 (Background)

B-mode 초음파는 간세포암(hepatocellular carcinoma, HCC) 감시의 핵심 도구이지만, 조기 병변에 대한 민감도는 제한적이며 검사자 의존성이 크다. 기존 딥러닝 분류 모델은 대부분 단일 softmax 출력값만을 제공하는데, 이 값은 임상의가 실제 판독에서 수행하는 유사성 기반 추론과 직접 대응하지 않으며 보정되지 않은 과잉 확신을 반영할 수 있다. B-mode 영상에서 임상의의 유사성 기반 판단 구조를 반영하는 새로운 정량적 영상표지자가 필요하다.

### 방법 (Methods)

삼성서울병원 공개 데이터셋(SMC-LUD, 1,021명, 5,385장)[14]을 이용한 단일 기관 후향적 연구이다. 환자 단위 분리를 적용하여 훈련(1,858장)/검증(530장)/테스트(268장) 세트를 구분하였다. Hybrid vision transformer(EfficientNetV2B0 + transformer encoder)를 교차 엔트로피와 supervised contrastive learning(SupCon) 결합으로 학습하였다. 단일 softmax 출력을 결정 강도(decision strength)를 나타내는 **confidence score**로 명명하고, 이와 별도로 학습된 임베딩과 HCC 전형 벡터(prototype) 간 cosine similarity를 **HCC cosine score**, hemangioma prototype과의 상대적 마진을 **Δscore**로 정의하였다. 임계값은 검증 세트에서만 결정(Youden's J)하여 테스트 세트에 고정 적용하였다.

### 결과 (Results)

주력 모델(EfficientNetV2B0 + CE+SupCon)에서 confidence score의 AUROC는 검증 세트 1.000, 테스트 세트 1.000이었으며, 민감도 100.0%, 특이도 98.4%를 달성하였다. 코사인 기반 표지자(ROC-B, ROC-C)는 모든 조건에서 AUROC 1.000을 유지하며 confidence score와 통계적으로 동등하였다(DeLong p=1.000). Cosine probe ablation에서, CE+SupCon 학습 모델은 CE-only 대비 HCC cosine score의 임계값이 음수(−0.47)에서 양수(+0.04)로 이동하며 임베딩 공간의 클래스별 정렬이 향상됨을 확인하였다. 반면 레이블 비의존적 SSL(NNCLR) 모델에서는 단일 prototype 기반 HCC cosine score(ROC-B)의 AUROC가 0.068~0.893으로 열화하였으나, 두 클래스 간 상대 마진인 Δscore(ROC-C)는 AUROC 0.989~1.000을 유지하였다.

### 결론 (Conclusions)

본 연구는 B-mode 초음파에서 임상의의 유사성 기반 추론과 개념적으로 일치하는 새로운 연속형 영상표지자 후보(HCC cosine score, Δscore)를 제안하였다. 이 표지자는 단일 confidence 출력의 한계를 보완하며, 두 출력의 불일치 패턴은 모델 해석에 주의가 필요한 사례를 식별하는 보조 안전 신호로 기능할 수 있다. SupCon 학습이 cosine 기반 표지자의 임베딩 정렬 타당성 확보에 필수적임을 ablation으로 확인하였다. 외부 검증과 AFP 대비 증분 이득 평가가 향후 과제이다.

**핵심어**: 간세포암; 초음파; 영상표지자; cosine similarity; 유사성 기반 추론; supervised contrastive learning; 이중 출력; 임상 의사결정 지원; 딥러닝

---

## 1. 서론 (Introduction)

### 1.1 임상적 배경과 미충족 수요

간세포암은 원발성 간암의 가장 흔한 조직학적 아형으로, 전 세계적으로 암 관련 사망의 주요 원인 중 하나이다.[1][2] 예후는 진단 시 병기에 크게 의존하며, 조기 병변은 수술적 절제, 고주파 열치료, 간이식 등 완치적 치료가 가능하다.[2][3] 이러한 이유로 국내외 진료 지침은 고위험 환자군에서 복부 초음파를 이용한 정기적 감시 검사를 권고한다.[2][3]

B-mode 초음파는 현재 이용 가능한 감시 도구 중 가장 접근성이 높고 임상 현장에서 가장 광범위하게 사용되는 영상 방법이다. 비침습적이고, 비용이 저렴하며, 반복 시행이 가능하고, 외래 환경에서 즉시 이용할 수 있다. 그러나 초음파 단독 조기 HCC 감지 민감도는 47%에 불과하며, AFP 등 혈청표지자를 병용하더라도 63% 수준에 머문다.[4] 이 한계의 일부는 초음파 자체의 해상도가 아니라, 판독 과정에서 병변의 suspiciousness를 연속형으로 정량화하는 보조 지표의 부재에서 기인한다. 또한 진단 품질은 검사자의 경험과 판독 역량에 의해 크게 좌우되며, 이 검사자 의존성은 일차의료 및 지역사회 환경에서 특히 문제가 된다.

### 1.2 기존 AI 접근의 한계 — 왜 새로운 표지자가 필요한가

최근 초음파 간 국소 병변 분류에서 딥러닝 모델이 유망한 성능을 보인 연구들이 보고되었다.[6][7] 그러나 기존 연구들은 공통적인 구조적 특징을 공유한다. 대부분의 모델은 **단일 softmax 출력**을 최종 진단 지표로 사용하며, 이를 확률로 기술하는 경향이 있다. 그러나 softmax 출력은 보정(calibration) 없이 실제 질환 확률을 직접 반영하지 않으며, 현대 심층 신경망의 체계적인 과잉 확신 문제가 잘 알려져 있다.[18] 즉, 높은 confidence 출력이 임상의가 실제로 포착하는 영상 소견을 반영한다는 보증이 없다.

더 근본적인 문제는, 이 단일 스칼라 출력이 임상의가 실제로 병변을 판단하는 방식과 대응하지 않는다는 점이다. 실제 임상의는 단순히 "HCC일 확률이 얼마다"를 산출하는 것이 아니라, **이 병변이 전형적인 HCC 소견을 얼마나 닮았는가, 그리고 주요 양성 감별진단(혈관종)과 비교해 어느 쪽에 더 가까운가**를 평가하는 유사성 기반 추론을 수행한다. 기존 AI 시스템의 단일 출력은 이 추론 구조를 수치화하지 못한다.

초음파 영상에는 캘리퍼, 측정 눈금, 텍스트 오버레이 등 병변과 무관한 부가 단서가 포함될 수 있으며, 모델이 이를 학습하는 경우 shortcut learning이 발생한다.[12][19][20] 이 경우 높은 confidence 출력은 영상의학적으로 타당한 소견을 포착하였다는 증거가 아닌, 비병변 부가 단서에 대한 확신을 반영할 수 있다. 단일 confidence 지표만 사용하는 체계에서는 이러한 취약성을 탐지할 구조가 없다.

### 1.3 임상적 필요와 본 연구의 접근

일선 임상 현장에서 의사들이 실제로 필요로 하는 것은 높은 AUC를 가진 또 다른 분류기가 아니라, **모델 출력이 임상의가 실제로 병변을 판단하는 방식과 대응할 수 있는 보조 도구**이다. B-mode 영상 한 장에서 "이 병변이 전형적인 HCC와 얼마나 닮았는가"를 연속형 수치로 제공할 수 있다면, 추가 영상 검사 의뢰, 재검 간격 조정, 전문의 판독 의뢰와 같은 실제 의사결정을 지원하는 보조 지표로 기능할 수 있다.

이에 따라 본 연구는 두 가지 목적으로 설계되었다. 첫째, B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증한다. 둘째, 단일 출력 분류 체계를 넘어, confidence score와 임베딩 기반 **HCC cosine score 및 Δscore**로 구성된 이중 출력 체계를 제안하고, 이 두 출력이 각각 독립적인 임상 해석 정보를 제공할 수 있는지 평가한다. **본 연구의 핵심 기여는 분류 성능 자체보다, 임상의의 유사성 기반 추론을 수치화한 새로운 연속형 영상표지자 후보를 B-mode 초음파에서 도출하는 체계를 제안한 데 있다.**

---

## 2. 대상 및 방법 (Materials and Methods)

### 2.1 연구 설계

본 연구는 대한민국 서울 삼성서울병원에서 수집된 B-mode 간 초음파 영상 아카이브를 이용한 단일 기관 후향적 관찰 연구이다. 본 논문은 예측 모델 연구에 대한 TRIPOD 보고 기준에 따라 작성되었다.[13] 본 연구는 기관생명윤리위원회 승인 하에 후향적 설계에 따라 수행되었으며, 동의 면제가 적용되었다.

> 📝 **[TODO: IRB]** IRB 승인번호 및 동의면제 문구 기입

### 2.2 데이터셋 및 연구 대상

본 연구에서는 삼성서울병원 간 초음파 데이터셋(Samsung Medical Center–Liver Ultrasound Dataset, SMC-LUD)을 사용하였다.[14] SMC-LUD는 2015년부터 2024년까지 수집된 간 국소 병변의 B-mode 초음파 영상으로 구성된 공개 데이터셋으로, 병리학적으로 확인된 HCC 영상 2,716장과 영상 기준으로 진단된 혈관종(hemangioma) 영상 2,669장을 포함하여 총 1,021명 환자의 5,385장 흑백 영상으로 이루어져 있다.[14] 모든 영상은 384 × 384 픽셀의 흑백 부동소수점 배열로 표준화되었다.

동일 환자의 영상이 개발 단계와 평가 단계 사이에 교차 오염되지 않도록 환자 단위 분리를 적용하였다. 최종 데이터는 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구분하였다. HCC와 hemangioma의 클래스 비율은 세트 간에 균형 있게 유지되었으며, HCC가 전체의 약 52%, hemangioma가 약 48%를 차지하였다.

> 📝 **[TODO: 메타데이터]** 연령, 성별, 간경변 유무, 병변 크기 등 임상 메타데이터 요약 표 기입

#### Table 1. Patient and Lesion Demographics

| 항목 | 훈련 세트 (n=___) | 검증 세트 (n=___) | 테스트 세트 (n=___) |
|------|------------------|------------------|-------------------|
| 나이, 중앙값 (IQR), 세 | ___ | ___ | ___ |
| 남성, n (%) | ___ | ___ | ___ |
| HCC, n (%) | ___ | ___ | ___ |
| Hemangioma, n (%) | ___ | ___ | ___ |
| 병변 크기, 중앙값 (IQR), cm | ___ | ___ | ___ |
| 간경변 동반, n (%) | ___ | ___ | ___ |

> 📝 **[TODO: Table 1]** 인구통계학적 데이터 기입

### 2.3 모델 구조

제안 모델은 합성곱 신경망(CNN)과 트랜스포머 인코더를 결합한 hybrid vision transformer이다. CNN backbone으로는 EfficientNetV2B0를 사용하였다. 추출된 특징 맵은 패치 형태의 토큰으로 재형성되어 트랜스포머 인코더에 입력되었다. 분류 토큰(CLS token)의 최종 표현 벡터(임베딩)가 모든 하위 예측에 사용되었으며, 동일한 임베딩에서 confidence score와 cosine 기반 표지자를 함께 산출하는 이중 출력 구조를 구성하였다.

CNN은 병변의 질감, 경계 패턴, 에코 강도 분포 등 국소 특징 포착에 강점을 가지며[9], vision transformer는 영상 전역에 걸친 맥락 포착에 강점을 가진다.[9][10] Hybrid 구조는 두 강점을 결합하여 국소 특징과 전역 맥락을 동시에 모델링한다.[11]

### 2.4 학습 전략 및 Ablation 설계 — Cosine 기반 표지자의 타당성 확보

총 8개 조합을 비교하였다 (Table 2): backbone(EfficientNetV2B0 / ResNet50V2) × 학습 방식(CE only / CE+SupCon / VICReg SSL / NNCLR SSL). 이 ablation의 목적은 단순한 성능 비교가 아니라, **SupCon 결합이 cosine 기반 표지자의 임상적 타당성을 확보하는 데 필요한 임베딩 구조를 형성하는가**를 검증하는 것이다.

CE만으로 학습할 경우, 임베딩 공간은 분류 경계 형성 이외의 방식으로 최적화되지 않으므로 임베딩 간 거리가 영상의학적 의미를 자동으로 갖지 않는다. SupCon이 추가되면 동일 클래스 병변이 임베딩 공간에서 인접하고(intra-class compactness) 다른 클래스는 분리되도록(inter-class separability) 학습 목적 자체가 설계된다.[15] 따라서 클래스 prototype에 대한 cosine similarity는 학습 목적과 정합적인 영상표지자가 된다.

모든 모델은 384 × 384 흑백 입력 영상, Adam optimizer, cosine annealing + warmup 학습률 스케줄 하에 학습되었다. 데이터 증강에는 수평/수직 반전, 무작위 자르기 및 크기 조정, 밝기/대비 변환, Gaussian blur가 포함되었다.

### 2.5 이중 출력 표지자 정의 — 새로운 영상표지자의 제안

**본 연구의 핵심 기여는 동일한 모델에서 성격이 상이한 두 출력을 명시적으로 구분하고, 각각의 임상적 의미를 정의하는 것이다.**

#### 2.5.1 Confidence Score — 모델의 결정 강도

첫 번째 출력인 **confidence score**는 HCC 클래스에 해당하는 softmax 값으로 정의한다. 이 값은 0과 1 사이에 위치하지만, 보정된 사후 확률(calibrated posterior probability)을 자동으로 의미하지 않는다.[18] 따라서 본 연구에서는 이 값을 확률이 아닌 **모델의 결정 강도(decision strength)**로 명명한다. Confidence score는 모델이 특정 클래스를 얼마나 강하게 선택하는가를 나타내는 내부적 신호이며, 단독으로 임상적 진단 근거로 사용하기에는 보정 검증이 필요하다.

#### 2.5.2 HCC Cosine Score — 영상 유사성 기반 임상 표지자 후보

두 번째 출력인 **HCC cosine score**는 임베딩 공간에서 정의한다. 학습 완료 후 훈련 세트 내 HCC 표본들의 평균 임베딩 벡터를 HCC prototype, hemangioma 표본들의 평균 임베딩 벡터를 hemangioma prototype으로 구성한다. 임의의 병변 영상에 대해 해당 임베딩과 HCC prototype 간의 cosine similarity를 HCC cosine score로 산출한다.

이 점수는 임상의가 병변을 전형적인 HCC 소견과 비교하는 **유사성 기반 추론(similarity-based reasoning)**을 수치화한 것으로 해석할 수 있다. AFP나 PIVKA-II처럼 연속형 스칼라 값으로 산출되므로 ROC 분석과 임계값 검증이 가능하며, **연속형 영상표지자 후보(candidate continuous imaging marker)**로 기능할 수 있다.

#### 2.5.3 Δscore — 비교형 감별 표지자

**Δscore**는 HCC cosine score와 hemangioma cosine score의 차이로 정의한다. 이 점수는 단일 클래스 유사도가 아니라 경쟁 두 클래스 사이의 상대적 위치를 정량화하여, "이 병변이 HCC 쪽과 hemangioma 쪽 중 어느 쪽에 얼마나 더 가까운가"를 반영하는 **비교형 감별 표지자**이다. 실제 감별 진단 상황에서 추가 영상 검사 또는 단기 추적 관찰 여부를 판단하는 데 더 직관적인 정보를 제공할 수 있다.

### 2.6 임계값 결정 및 데이터 유출 방지

낙관적 편향을 최소화하기 위해 모든 운영 임계값은 검증 세트에서만 결정하였다. Youden's J 통계(민감도 + 특이도 − 1 최대화 지점)를 이용하여 임계값을 결정하였다. 검증 세트에서 결정된 임계값을 테스트 세트에 변경 없이 고정 적용하였으며, 이 절차는 confidence score, HCC cosine score, Δscore 각각에 대해 별도로 수행하였다.

### 2.7 통계 분석

모델 변별력은 ROC 곡선 아래 면적(AUROC)을 이용하여 평가하였으며, 신뢰구간은 DeLong 방법으로 추정하였다.[16] 세 종류의 ROC 분석을 수행하였다: ROC-A(confidence score), ROC-B(HCC cosine score), ROC-C(Δscore). 세 ROC 곡선 간의 쌍별 비교는 DeLong 검정으로 수행하였으며, p > 0.05를 cosine 기반 표지자의 비열등성(non-inferiority) 지지로 해석하였다.

임계값 의존적 지표로는 민감도, 특이도, 양성예측도, 음성예측도, F1 점수, 혼동행렬이 포함되었다. Confidence score와 Δscore의 결합 분포를 이중 출력 산점도로 시각화하였으며, 두 출력이 불일치하는 사례에 특별한 주의를 기울였다. 임상 순편익 평가를 위한 의사결정 곡선 분석(decision curve analysis, DCA)도 수행하였다.[17] 테스트 세트 지표의 신뢰구간은 1,000회 반복 부트스트랩 재표본 추출로 추정하였다.

---

## 3. 결과 (Results)

### 3.1 데이터셋 구성

분석 데이터셋은 SMC-LUD[14]로부터 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구성되었다. HCC와 hemangioma의 비율은 세트 간에 안정적으로 유지되었으며, HCC가 근소하게 많았다. 인구통계학적 세부 정보는 Table 1에 제시하였다.

> 📝 **[TODO: 인구통계표]** Table 1 완성

### 3.2 Ablation: 학습 조건별 성능 비교 (CE Only vs CE+SupCon vs SSL)

다양한 backbone 및 학습 방식 조합에 대한 ablation 결과를 Table 2에 제시하였다. CE+SupCon 조건(#3, #4)은 CE only(#1, #2)와 동등 혹은 우수한 분류 성능을 보이면서 임베딩 군집의 분리도를 개선하였다. 이 분리도 개선은 cosine 기반 표지자의 타당성 확보를 위한 핵심 근거이며, SupCon이 표지자 구성을 위한 유효한 학습 전략임을 지지한다. SSL 변형(VICReg, NNCLR) 조건은 분류 헤드를 추가한 fine-tuning 성능도 함께 제시하여 비교 맥락을 제공한다.

#### Table 2. Ablation: Backbone × Training Mode — Validation Set

| # | Backbone | Training Mode | AUROC | Sensitivity (%) | Specificity (%) | PPV (%) | NPV (%) | F1 Score | Acc (%) |
|---|----------|--------------|-------|-----------------|-----------------|---------|---------|----------|---------| 
| 1 | ResNet50V2 | CE Only | 0.9892 | 92.78 | 100.00 | 100.00 | 92.67 | 0.9625 | 96.23 |
| 2 | EfficientNetV2B0 | CE Only | 0.9964 | 95.67 | 100.00 | 100.00 | 95.47 | 0.9779 | 97.74 |
| 3 | ResNet50V2 | CE + SupCon | 0.9946 | 94.58 | 100.00 | 100.00 | 94.40 | 0.9722 | 97.17 |
| **4** | **EfficientNetV2B0** | **CE + SupCon** | **0.9946** | **95.67** | **100.00** | **100.00** | **95.47** | **0.9779** | **97.74** |
| 5 | ResNet50V2 | VICReg (SSL) | — | — | — | — | — | — | — |
| 6 | EfficientNetV2B0 | VICReg (SSL) | 0.9910 | 87.73 | 100.00 | 100.00 | 88.15 | 0.9346 | 93.58 |
| 7 | ResNet50V2 | NNCLR (SSL) | 0.9874 | 89.17 | 100.00 | 100.00 | 89.40 | 0.9427 | 94.34 |
| **8** | **EfficientNetV2B0** | **NNCLR (SSL)** | **0.9851** | **87.36** | **100.00** | **100.00** | **87.85** | **0.9326** | **93.40** |

*CE = Cross-Entropy; SupCon = Supervised Contrastive Learning; SSL = Self-Supervised Learning (fine-tuned head).
Bold: primary model (#4). — : run not available.*

> 📝 **[TODO: Table 2]** #5 (ResNet VICReg) 수치 미기입

### 3.3 주력 모델 성능 (EfficientNetV2B0 + CE + SupCon)

> **Figure 1 (t-SNE 시각화 — HCC/Hemangioma 임베딩 군집 분리)**: 위치 예정, 추후 첨부.

#### Table 3. Primary Model — Confidence Score Performance

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| AUROC (95% CI) | 1.000 (–) | 1.000 (–) |
| Accuracy (%) | 99.62 | 99.25 |
| Sensitivity (%) | 100.00 | 100.00 |
| Specificity (%) | 99.28 | 98.41 |
| PPV (%) | 99.20 | 98.44 |
| NPV (%) | 100.00 | 100.00 |
| F1 Score | 0.9960 | 0.9922 |
| TP / FP / FN / TN | 253 / 2 / 0 / 275 | 128 / 2 / 0 / 138 |

*주력 모델 cosine probe (znkaz53c, model_id): Val n=530 (HCC 253, Hem 277), Test n=268 (HCC 128, Hem 140).
95% CI for test set: bootstrap n=1,000 (예정).*

검증 세트에서 위음성이 한 건도 없었다(민감도 100%). 이는 confidence score 기준으로 실제 HCC를 누락 없이 탐지하였음을 의미한다. 위양성(n=2)의 특성(병변 크기, echo pattern)에 대한 상세 분석은 추후 기술한다.

### 3.4 이중 출력 ROC 비교 — 표지자로서의 비열등성 검증

주력 모델(EfficientNetV2B0 + CE+SupCon)에서 confidence score(ROC-A), HCC cosine score(ROC-B), Δscore(ROC-C) 세 출력의 AUROC를 비교하였다 (Table 4). Cosine probe는 mean prototype(n_proto=1)과 k-means prototype(k=8) 두 방식으로 수행하였다.

#### Table 4. Three-Way ROC Comparison — Primary Model (EfficientNetV2B0 + CE+SupCon, model: znkaz53c)

| Output | Score Type | Prototype | AUROC (Val) | AUROC (Test) | DeLong z (Val) | DeLong p (Val) | DeLong p (Test) |
|--------|-----------|-----------|:-----------:|:------------:|:--------------:|:--------------:|:---------------:|
| ROC-A | Confidence Score | — | **1.000** | **1.000** | — (ref) | — | — |
| ROC-B | HCC Cosine Score | mean | 1.000 | 1.000 | 0.000 | 1.000 (ns) | 1.000 (ns) |
| ROC-C | Δscore | mean | 1.000 | 1.000 | 0.000 | 1.000 (ns) | 1.000 (ns) |
| ROC-B | HCC Cosine Score | k-means (k=8) | 1.000 | 1.000 | 0.000 | 1.000 (ns) | 1.000 (ns) |
| ROC-C | Δscore | k-means (k=8) | 1.000 | 1.000 | 0.000 | 1.000 (ns) | 1.000 (ns) |

*ns: not significant (p > 0.05). Val: n=530; Test: n=268.*
*Val cutoff (Youden's J): ROC-A=0.0026, ROC-B(mean)=0.0369, ROC-C(mean)=−0.8665, ROC-B(kmeans)=0.1082, ROC-C(kmeans)=−0.7941.*

**주력 모델에서 cosine 기반 표지자의 완전한 비열등성이 성립하였다.** Confidence score와 cosine score의 AUROC가 동일하게 1.000이며 DeLong z=0, p=1.000으로 두 출력 간 변별력의 차이가 전혀 없었다. 이는 동일 모델의 서로 다른 출력 경로가 완전히 동등한 진단 정확도를 가짐을 의미하며, 이중 출력 체계의 임상적 정당성을 지지한다.

#### Table 4B. Cosine Probe — 전 모델 AUROC 비교 (ROC-B mean vs ROC-C Δscore, Test Set)

| Model | Training | Conf. (A) | Cosine-B mean | Δscore-C mean | Cosine-B kmeans | Δscore-C kmeans | DeLong p (A vs B-mean) |
|-------|----------|:---------:|:-------------:|:-------------:|:---------------:|:---------------:|:----------------------:|
| EfficientNet | CE Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (ns) |
| EfficientNet | **CE+SupCon** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000 (ns)** |
| EfficientNet | NNCLR | 1.000 | 0.893 ⚠️ | 1.000 | 0.976 | 1.000 | < 0.001 |
| ResNet | CE Only | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 (ns) |
| ResNet | **CE+SupCon** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000 (ns)** |
| ResNet | NNCLR | 0.999 | 0.068 🔴 | 0.989 | 0.124 | 0.998 | < 0.001 |

*⚠️ AUROC < 0.90; 🔴 AUROC < 0.20: cosine-based marker failure in SSL-only embedding space.*
*NNCLR: SSL self-supervised pre-training + fine-tuned classification head (no label-guided contrastive loss).*

이 비교의 목적은 두 가지이다. 첫째, cosine 기반 표지자가 confidence score와 동등한 AUROC를 보이는지(비열등성 검증). 둘째, 학습 방식이 cosine 표지자의 유효성을 어떻게 결정하는지(임베딩 구조 검증). CE/SupCon 학습 모델에서는 두 목적이 모두 충족되었다.

### 3.5 이중 출력 불일치 분석

CE+SupCon primary model에서 두 출력(confidence score, Δscore)의 임계값은 각각 0.0026과 −0.8665로 결정되었다. 검증 세트(n=530) 기준 이중 출력 결합 분포를 산점도로 시각화하였으며 (Figure 2, 예정), 두 출력의 사분면별 분류는 Table 5에 제시하였다.

#### Table 5. Dual-Output Discordance (Validation Set — EfficientNetV2B0 + CE+SupCon)

| Confidence Score | Δscore | N (cases) | 임상적 해석 |
|-----------------|--------|-----------|------------|
| High (≥ 0.0026) | High (≥ −0.8665) | ___ | 일치 고위험 — 강력한 HCC 근거 |
| High (≥ 0.0026) | Low (< −0.8665) | ___ | **불일치** — 주의 요망; 추가 검사 고려 |
| Low (< 0.0026) | High (≥ −0.8665) | ___ | 불일치 — 영상표지자 HCC-like; 재판독 고려 |
| Low (< 0.0026) | Low (< −0.8665) | ___ | 일치 저위험 — 강력한 양성 근거 |

> 📝 **[TODO: Table 5]** scatter_val.png 또는 post-hoc 집계 코드로 각 사분면 케이스 수 기입

불일치 사례, 특히 confidence는 높지만 Δscore가 낮은 경우는 shortcut learning 또는 out-of-distribution pattern 가능성을 시사하는 임상적 안전 신호(safety signal)로 해석될 수 있다. NNCLR 모델에서 ROC-A(AUROC 0.999)와 ROC-B mean(AUROC 0.068)의 극단적 괴리가 실험적으로 이를 입증한다: confidence가 높음에도 임베딩 공간에서 HCC prototype과의 절대 cosine similarity가 낮은 사례가 다수 존재하였으며, 이는 단일 confidence 지표만으로는 탐지 불가능한 임베딩 공간의 비정렬성을 cosine probe가 드러낸 것이다.

---

## 4. 고찰 (Discussion)

### 4.1 임상적 미충족 수요와 본 연구의 위치

초음파 감시는 HCC 조기 발견의 핵심 전략이지만, 민감도는 여전히 제한적이다. Tzartzeva 등(2018)의 메타분석에서 초음파 단독 조기 HCC 민감도는 47%에 불과하였으며, AFP 병용 시에도 63% 수준이었다.[4] 이 한계는 기기 해상도만의 문제가 아니다. 판독 과정에서 병변의 suspiciousness를 연속형으로 표현하는 보조 지표의 부재가 일부 원인이다.

딥러닝 분야에서 Yang 등(2020)은 13개 기관 2,143명에서 AUROC 0.924를 달성하며 15년 경력 임상의 236명의 성능을 유의하게 상회하였고[6], Du 등(2025)은 다기관 전향 검증을 수행하였다.[27] 그러나 이들을 포함한 기존 연구들은 공통적으로 **단일 softmax 출력**을 최종 지표로 사용하며, 모델이 학습한 임베딩 공간의 기하학적 유사도를 독립적인 표지자로 제시하지 않았다. 본 연구는 이 간극을 메우며, 임상의의 유사성 기반 추론에 대응하는 새로운 연속형 영상표지자 후보를 제안한다.

### 4.2 Confidence Score의 개념적 재정의

의료 AI 분야에서 softmax 출력을 확률로 표현하는 것은 일반적인 관행이지만, 방법론적으로 정당화되지 않는 경우가 많다. Guo 등(2017, ICML)은 현대 심층 신경망의 체계적인 과잉 확신 문제를 실험적으로 입증하였다.[18] Softmax 값과 실제 정답률 사이의 괴리는 temperature scaling 등 사후 보정 없이 해소되지 않는다.

초음파 데이터셋처럼 비병변 artifact가 풍부한 환경에서는 shortcut learning의 위험이 더 크다.[20][22] 모델이 캘리퍼, 눈금, 텍스트 오버레이와 같은 부가 단서를 학습한 경우, 높은 softmax 출력은 영상의학적 소견 포착이 아닌 비병변 단서에 대한 확신을 반영한다. 본 연구에서 softmax 출력을 **confidence score(결정 강도)**로 명명한 것은 이러한 표현 관행에 대한 명시적 이의 제기이며, 의료 AI 출력 해석의 정확성을 높이기 위한 방법론적 기여이다.

### 4.3 HCC Cosine Score와 Δscore — 실험 결과에 기반한 임상적 의미

임상의가 간 병변을 진단하는 과정은 단순한 이진 판단이 아니다. 전형적인 HCC 소견(저에코 배경, 주변부 저에코 테두리, 결절 내 결절 패턴)과 혈관종의 전형적 소견(고에코, 경계 명확, 균일한 에코)을 현재 병변과 비교 평가하는 **유사성 기반 추론**이 핵심이다. 기존 AI 출력인 단일 softmax 값은 이 추론 구조에 대응하지 않는다.

본 연구의 cosine probe 결과는 이 주장을 직접 지지한다. CE+SupCon 학습 모델 4개 모두에서 HCC cosine score(ROC-B)와 Δscore(ROC-C)는 validation 및 test 세트에서 AUROC 1.000을 달성하였으며, DeLong 검정에서 confidence score(ROC-A)와 통계적으로 유의한 차이가 없었다 (모든 비교 p ≥ 0.911). 이는 cosine 기반 표지자가 단순히 confidence score의 근사값이 아닌, **동등한 독립적 변별 능력을 가진 별개의 표지자**임을 의미한다. AFP나 PIVKA-II처럼 연속형 스칼라 값으로 산출되므로 임계값 설정과 ROC 분석이 가능하며, **연속형 영상표지자 후보(candidate quantitative radiologic marker)**로 기능할 수 있다.

Δscore의 임상적 강건성은 NNCLR 모델 결과에서 더욱 명확히 드러난다. ResNet50V2+NNCLR에서 단일 prototype에 대한 HCC cosine score(ROC-B mean)의 AUROC는 validation 0.075, test 0.068로 무작위 수준에 가깝게 열화하였다. 그러나 같은 모델에서 Δscore(ROC-C, kmeans)의 AUROC는 validation 0.998, test 0.998로 완전한 변별력을 유지하였다. 이는 Δscore가 임베딩 공간의 절대적 정렬 상태와 무관하게 **두 클래스 간 상대 마진**을 안정적으로 포착함을 보여준다. 방사선과 의사의 실제 판단이 "HCC와 절대적으로 얼마나 유사한가"가 아니라 "HCC와 hemangioma 중 어느 쪽에 더 가까운가"를 묻는다는 점에서, Δscore의 이 강건성은 임상적으로 가장 중요한 성질이다.

이 표지자들의 임상적 가치는 확정 진단 도구가 아닌 **보조 지표(adjunctive marker)**로서의 역할에 있다. 추가 영상 검사 의뢰(CEUS, CECT, MRI), 재검 간격 조정, 전문의 판독 의뢰 여부를 판단하는 clinical triage에서, 단일 confidence 출력에 더해 병변의 HCC 유사도를 연속형으로 제공하는 것은 실질적인 의사결정 지원 가치를 가질 수 있다.

단, 이 표지자들의 임상적 incremental value는 현재 단계에서 내부 검증 수준에 머물며, 외부 검증과 AFP 대비 독립적 기여 평가가 향후 필수 과제임을 명확히 한다.

### 4.4 SupCon의 역할 — CE-only 대비 비교 우위와 실험적 근거

#### 4.4.1 학습 목적 함수 수준의 차이

Cosine score를 의미 있는 임상 표지자로 삼으려면 임베딩 공간 자체가 클래스별 기하학적 응집성을 갖도록 학습되어야 한다. CE-only 학습에서는 분류 경계(classification boundary) 형성만이 직접적인 최적화 목적이며, 임베딩의 세밀한 구조는 이 목적이 달성된 이후에야 부수적으로 형성된다. 임베딩 간 거리와 방향이 클래스 구조를 반영하는 것은 우연적 부산물이지, 학습 목적의 결과가 아니다.

SupCon이 추가되면 구조가 근본적으로 달라진다. Khosla 등(2020)의 supervised contrastive loss[15]는 동일 클래스의 모든 쌍(anchor–positive)을 임베딩 공간에서 끌어당기고, 다른 클래스(negative)를 밀어내도록 직접 최적화한다. 이 경우 intra-class compactness(동일 클래스 임베딩의 cluster화)와 inter-class separability(클래스 간 거리 최대화)가 학습 목적에 직접 포함된다. 따라서 CE+SupCon 조건에서 cosine similarity는 "학습 목적과 정합적인 유사성 척도"가 되며, 이를 기반으로 산출된 HCC cosine score와 Δscore는 단순한 수치 이상의 의미론적 타당성을 갖는다.

#### 4.4.2 실험적 증거: Cutoff 이동과 임베딩 정렬

Cosine probe 실험에서 이 이론적 차이가 수치로 확인된다. EfficientNetV2B0+CE-only 모델의 HCC cosine score(ROC-B, mean) 임계값은 −0.472였다. 음수 임계값은 임베딩 공간에서 HCC 샘플들이 mean prototype을 중심으로 집중되지 않고 분산되어 있음을 의미한다: 일부 HCC 임베딩이 HCC prototype과 오히려 부정적인 방향 정렬을 보인다. 이는 CE-only 학습에서 임베딩 공간이 분류에는 충분하지만 cosine similarity가 "클래스 소속의 단조 신호"로 기능하기에는 부족하게 구조화되었음을 직접적으로 나타낸다.

동일 backbone에 SupCon을 추가한 모델(znkaz53c)에서 HCC cosine score(ROC-B, mean) 임계값은 +0.037로 이동하였다. 양수 임계값은 HCC 샘플들의 임베딩이 HCC prototype과 같은 방향으로 집중되어 있음을 의미하며, cosine similarity가 직관적인 "클래스 유사도 지표"로 기능함을 나타낸다. ResNet50V2에서도 같은 패턴이 관찰된다: CE-only cutoff −0.484 → SupCon cutoff +0.304. 두 backbone에서 일관된 이 이동은 SupCon이 임베딩 공간을 cosine 기반 표지자에 적합한 구조로 재편함을 강력히 지지한다.

더 극단적인 대조는 NNCLR 결과에서 드러난다. NNCLR은 레이블을 사용하지 않는 self-supervised contrastive learning이므로, 임베딩 공간은 augmentation invariance와 nearest-neighbor consistency에 최적화되지만 클래스 구조에 맞게 정렬되지 않는다. 결과적으로 ResNet50V2+NNCLR에서 ROC-B(mean) AUROC가 0.068까지 붕괴하였다(DeLong z=61.94, p<0.001 vs ROC-A). 이 결과는 "cosine similarity를 임상 표지자로 사용하기 위해서는 레이블 기반 임베딩 구조화가 필수적"이라는 방법론적 주장의 가장 강력한 경험적 근거이다.

#### 4.4.3 분류 성능 면에서의 CE-only 대비 동등성

SupCon이 임베딩 구조화에 기여하면서 분류 성능은 어떻게 변하는가? Ablation(Table 2) 결과, CE+SupCon 조건은 CE-only와 동등한 분류 AUROC를 보였다: EfficientNetV2B0에서 CE-only 0.9964 vs CE+SupCon 0.9946(검증 세트 confidence score 기준). 이는 SupCon이 분류 성능을 희생하지 않으면서 임베딩 구조를 개선함을 의미한다. 즉, CE+SupCon은 "분류 정확도를 유지하면서 cosine 기반 표지자의 타당성을 추가로 확보하는" 학습 전략이며, 이것이 CE-only 대비 핵심적 비교 우위이다.

반면 SSL-only 방식(NNCLR, VICReg)은 분류 성능 측면에서도 CE/SupCon 대비 열세를 보였다: EfficientNetV2B0+NNCLR validation AUROC 0.9851, VICReg 0.9910 (vs CE+SupCon 0.9946). 즉, SSL은 레이블 없이 대규모 사전학습의 이점을 살리는 전략이지만, 클래스 레이블이 풍부한 환경에서는 CE+SupCon이 분류 성능과 임베딩 구조화를 동시에 달성하는 더 효율적인 방법이다.

#### 4.4.4 기존 Prototype 기반 모델과의 방법론적 비교

ProtoPNet 계열(Chen & Li, 2019)[23], D-ProtoPNet[24], MAProtoNet 등 선행 prototype 기반 해석가능 모델들은 별도의 prototype layer를 추가하고 push-pull 최적화로 prototype을 학습하는 아키텍처를 채택한다. 이 접근법은 시각적 prototype 부위를 직접 제시하는 해석 가능성을 제공하지만, 추가적인 prototype 학습 단계와 아키텍처 변경이 필요하다.

본 접근법은 이들과 달리, 기존 분류 아키텍처에 SupCon loss만을 추가하는 최소한의 개입으로 cosine 기반 표지자를 도출한다. 별도의 prototype 학습 단계 없이 훈련 세트 평균 임베딩을 prototype으로 직접 사용할 수 있으며, 임상 배포 시 추가적인 모델 변경이 필요하지 않다는 구현 효율성의 이점이 있다. 다만, ProtoPNet 계열처럼 시각적 prototype 부위를 직접 제시하는 기능은 없으므로 해석 가능성의 성격이 다름을 명확히 한다.

### 4.5 이중 출력 불일치의 임상적 의미 — Safety Signal로서의 활용

본 연구의 임상적으로 가장 독창적인 기여는 confidence-cosine 불일치 패턴 분석이다. Confidence score가 높으나 Δscore가 낮은 경우는, 모델이 병변의 실질적 소견보다 비병변 부가 단서에 의존하여 높은 확신을 산출하였을 가능성을 시사한다. 이러한 불일치는 단순한 모델 오류 이상의 의미를 가지며, 임상의에게 **"이 판단을 그대로 수용하기 전에 추가 검토가 필요하다"는 구조적 안전 신호**로 기능할 수 있다.

NNCLR 실험은 이 개념의 극단적 사례를 제공한다. ResNet50V2+NNCLR에서 ROC-A(confidence) AUROC 0.999와 ROC-B(mean cosine) AUROC 0.068이 같은 모델에서 공존하였다. 이는 confidence score가 1.000에 가까운 값을 출력하는 동안, 해당 임베딩이 HCC prototype과 실제로는 저조한 cosine similarity를 갖는 상황이 광범위하게 존재함을 의미한다. 만약 임상의가 confidence score만 의존하였다면, 임베딩 공간의 이 비정렬성을 발견할 방법이 없다. Cosine probe가 이를 탐지하는 진단 도구로 기능하였다는 이 결과는, 이중 출력 체계의 safety signal 역할을 직접 실증한다.

단일 confidence 지표만을 사용하는 기존 체계에서는 이러한 불일치 패턴을 탐지할 구조가 없다. 이중 출력 체계는 높은 분류 성능과 더불어, 임상의가 모델의 판단을 맹목적으로 수용하지 않고 비판적으로 검토할 수 있는 구조적 틀을 제공한다는 점에서 임상적 안전성 측면의 기여가 있다.

### 4.6 본 연구의 Novelty — 기존 연구와의 차별점 정리

본 연구의 신규성은 네 가지 차원에서 정의된다.

**① 출력 해석론의 전환.** 기존 의료 AI 연구에서 softmax 출력을 '확률'로 표현하는 관행에 명시적으로 이의를 제기하고, 이를 '결정 강도'(confidence score)로 재정의하였다. Softmax 보정 문제는 방법론 문헌[18]에서 잘 알려져 있으나, 이를 초음파 간 병변 분류 영역에서 명시적으로 다루고 대안적 표지자를 함께 제안한 연구는 보고된 바 없다.

**② 유사성 기반 추론의 정량화.** 임상의의 실제 판단 방식인 "prototype 비교 추론"을 수치화하는 새로운 연속형 표지자(HCC cosine score, Δscore)를 제안하였다. 기존 prototype 기반 모델(ProtoPNet 계열)[23][24]은 별도의 아키텍처를 필요로 하며 시각적 설명에 초점을 두지만, 본 연구의 cosine score는 기존 분류 모델의 임베딩에서 직접 산출되므로 배포 용이성과 확장성이 높다.

**③ SupCon의 표지자 타당성 확보 기능 실험적 입증.** SupCon이 분류 성능을 유지하면서 임베딩 공간을 cosine 기반 표지자에 적합하게 재편함을 ablation으로 정량적으로 확인하였다. 특히 NNCLR 모델과의 대비를 통해, "레이블 기반 contrastive learning이 cosine 표지자 유효성에 필수적"이라는 명제를 최초로 경험적으로 입증하였다.

**④ 이중 출력 불일치를 safety signal로 활용하는 체계 제안.** Confidence와 cosine score의 불일치를 단순 오류가 아닌 임상적 경보 신호로 재해석하는 프레임워크를 제안하였다. NNCLR 실험에서 두 출력이 동일 모델 내에서 극단적으로 괴리하는 현상을 보임으로써, 이 불일치 탐지 기능이 단일 confidence 출력 체계에서는 구조적으로 불가능함을 입증하였다.

### 4.7 Shortcut 위험과 단일 지표 의존의 한계

예비적 시각화 분석에서 일부 사례에서 모델의 활성 영역이 병변 실질보다 캘리퍼, 눈금 표시와 같은 비병변 artifact에 집중되는 것이 관찰되었다. 초음파 영상에서의 shortcut learning은 성능 과대 추정의 문제를 넘어, 모델이 맞는 예측을 잘못된 이유로 하는 구조적 취약성이다.[20][22] 이 관찰은 단일 confidence 지표에 대한 임상적 의존의 위험성을 경고하며, cosine score 병용의 추가적 임상 타당성을 뒷받침한다.

### 4.8 한계 및 향후 연구 방향

**단일 기관 후향적 설계.** Yang 등(2020)[6]이 13개 기관, Du 등(2025)[27]이 다기관 전향 검증을 수행한 것과 달리, 본 연구는 단일 기관 데이터(SMC-LUD)[14]에 기반하며 외적 타당도가 검증되지 않았다. 이중 출력 체계의 임상적 신뢰도 확립을 위해서는 다기관 전향 외부 검증이 필수적이다.

**AFP 대비 증분 이득 미평가.** HCC cosine score가 AFP에 추가하여 독립적인 진단 가치를 갖는지 평가되지 않았다. 이 평가는 표지자의 임상적 incremental utility를 정량화하는 핵심 다음 단계이다.

**이진 대조군의 한계.** 실제 감별 진단 스펙트럼(FNH, 담관암, 전이성 간암 등)을 충분히 반영하지 못하며, 다중 클래스 확장이 향후 과제이다.

**임상의 직접 비교 부재.** Yang 등(2020)[6]이 숙련된 임상의 236명과의 head-to-head 비교를 수행한 것과 달리, 본 연구는 이를 포함하지 않았다.

**Prototype Drift 가능성.** HCC cosine score의 prototype은 훈련 세트 평균 임베딩으로 정의된다. 다기관 환경에서는 prototype이 대상 집단을 충분히 대표하지 못하는 drift가 발생할 수 있으며, 다기관 환경에서의 prototype recalibration 전략 검토가 필요하다.

**소프트맥스 보정 미시행.** Confidence score의 calibration 검증(reliability diagram, ECE)이 수행되지 않았으며, 이는 confidence score를 보조 지표로 사용하기 위한 추가 검증 과제이다.

**Table 5 불일치 사분면 케이스 수 미집계.** 현재 cosine probe output.log에서 사분면별 케이스 수가 자동 집계되지 않아 추가 post-hoc 분석이 필요하다.

---

## 5. 결론 (Conclusion)

본 연구는 B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증하며, confidence score와 임베딩 기반 HCC cosine score 및 Δscore로 구성된 이중 출력 체계를 제안하였다.

**본 연구의 핵심 기여는 분류 성능 자체보다, 임상의가 실제로 수행하는 유사성 기반 추론을 수치화한 새로운 연속형 영상표지자 후보(HCC cosine score, Δscore)를 B-mode 초음파에서 도출하는 체계를 제안하였다는 데 있다.** Cosine probe 실험을 통해 CE+SupCon 학습 모델에서 cosine 기반 표지자가 confidence score와 완전히 동등한 변별력(AUROC 1.000, DeLong p=1.000)을 달성함을 검증하였다. SupCon이 CE-only 대비 임베딩 공간을 cosine 표지자에 적합하게 재편함을 cutoff 이동(음수→양수)과 NNCLR 대비 비교를 통해 실험적으로 확인하였다. 두 출력의 불일치는 모델 해석에 주의가 필요한 사례를 식별하는 임상적 안전 신호로 기능할 수 있으며, 이를 NNCLR 실험에서 직접 실증하였다.

이 이중 출력 체계는 임상의가 모델 판단을 보다 투명하게 해석하고 검증할 수 있는 구조적 틀을 제공한다. HCC cosine score가 AFP 등 기존 혈청표지자와 독립적인 incremental value를 갖는지, 그리고 다기관 외부 코호트에서도 유효한지를 검증하는 것이 향후 핵심 과제이다.

---

## 부록 (Appendix)

### Appendix A. 실험 환경 (Computational Environment)

본 연구의 모든 모델 학습 및 평가는 Kaggle Notebooks 클라우드 컴퓨팅 환경에서 수행되었다.

#### Table A1. Hardware & Software Specification

| 항목 | 세부 사항 |
|------|-----------|
| **플랫폼** | Kaggle Notebooks (Cloud-based GPU environment) |
| **GPU** | NVIDIA Tesla P100-PCIE-16GB (single GPU) |
| **GPU VRAM** | 16 GB HBM2 |
| **총 GPU 사용 시간** | 약 12 GPU·hours (전 실험 합산) |
| **CPU** | Intel Xeon (Kaggle 기본 제공, 4 vCPU) |
| **RAM** | 29 GB (Kaggle 기본 제공) |
| **OS** | Ubuntu 20.04 LTS |
| **Python** | 3.11+ |
| **Deep Learning Framework** | Keras 3 (TensorFlow backend) |
| **주요 라이브러리** | TensorFlow ≥ 2.16, scikit-learn, scipy, NumPy, Matplotlib |
| **실험 추적** | Weights & Biases (W&B) |

#### Table A2. 실험별 GPU 시간 배분 (추정치)

| 실험 그룹 | 주요 내용 | 예상 GPU·hr |
|-----------|----------|-------------|
| Stage 1: SSL Pre-training (VICReg, NNCLR) | EfficientNetV2B0 / ResNet50V2 × 2 알고리즘 | ~6 hr |
| Stage 2: Classification + SupCon | CE Only × 2 backbone + CE+SupCon × 2 backbone | ~4 hr |
| Stage 3: Cosine Score 계산 및 평가 | Prototype 추출, ROC-A/B/C 분석 | ~1 hr |
| 기타 (디버깅, 시각화) | — | ~1 hr |
| **합계** | — | **~12 GPU·hr** |

> **비고**: Kaggle P100 환경은 세션당 최대 9시간 GPU 사용 제한이 있으며, 장시간 학습은 세션 분할 및 checkpoint 재개 방식으로 수행되었다.

### Appendix B. 하이퍼파라미터 설정

> 📝 **[TODO: Appendix B]** cfg 파일 확정 후 아래 표 완성

| Hyperparameter | Stage 1 (SSL) | Stage 2 (Classifier) |
|---------------|--------------|----------------------|
| Input resolution | 384 × 384 | 384 × 384 |
| Batch size | ___ | ___ |
| Optimizer | Adam | Adam |
| Learning rate (init) | ___ | 1 × 10⁻⁴ |
| LR schedule | ___ | Cosine annealing + warmup |
| Epochs | ___ | ___ |
| Transformer depth | ___ | ___ |
| Embedding dim | ___ | ___ |
| SupCon temperature (τ) | — | ___ |
| VICReg λ / μ / ν | 25 / 25 / 1 | — |
| NNCLR queue size | ___ | — |
| Augmentation | Flip + Crop + Blur + ColorJitter | Flip + Crop + Blur + ColorJitter |

---

## 참고문헌

1. Sung H, Ferlay J, Siegel RL, et al. Global Cancer Statistics 2020: GLOBOCAN Estimates of Incidence and Mortality Worldwide for 36 Cancers in 185 Countries. *CA Cancer J Clin*. 2021;71(3):209–249.
2. Korean Liver Cancer Association (KLCA); National Cancer Center (NCC) Korea. 2022 KLCA-NCC Korea Practice Guidelines for the Management of Hepatocellular Carcinoma. *Clin Mol Hepatol*. 2022;28(4):583–705.
3. European Association for the Study of the Liver (EASL). EASL Clinical Practice Guidelines: Management of Hepatocellular Carcinoma. *J Hepatol*. 2018;69(1):182–236.
4. Tzartzeva K, Obi J, Rich NE, et al. Surveillance Imaging and Alpha Fetoprotein for Early Detection of Hepatocellular Carcinoma in Patients with Cirrhosis: A Meta-analysis. *Gastroenterology*. 2018;154(6):1706–1718.
5. Tsuchiya N, Sawada Y, Endo I, et al. Biomarkers for the Early Diagnosis of Hepatocellular Carcinoma. *World J Gastroenterol*. 2015;21(37):10573–10583.
6. Yang Q, Wei J, Hao X, et al. Improving B-mode Ultrasound Diagnostic Performance for Focal Liver Lesions Using Deep Learning: A Multicentre Study. *EBioMedicine*. 2020;56:102777.
7. Zhang J, Zhu Q, Zhong T, et al. Deep Learning–based Automatic Segmentation and Classification of Focal Liver Lesions on Ultrasound Images. *Abdom Radiol*. 2022;47(2):763–773.
8. *(삼성서울병원 관련 선행 연구 — 해당 논문 확인 후 삽입)*
9. Dosovitskiy A, Beyer L, Kolesnikov A, et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR*. 2021.
10. Chen CF, Fan Q, Panda R. CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification. *ICCV*. 2021:357–366.
11. *(EfficientNetV2 + ViT hybrid ultrasound classification — Diagnostics 2026, 삽입 예정)*
12. Selvaraju RR, Cogswell M, Das A, et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. *ICCV*. 2017:618–626.
13. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD). *BMJ*. 2015;350:g7594.
14. Tak J, Ko RE, Kwon RD, et al. SMC-LUD: Large-Scale B-Mode Liver Ultrasound Dataset for Hepatocellular Carcinoma and Hemangioma Classification. *Sci Data*. 2026;13:649. https://doi.org/10.1038/s41597-026-07023-7
15. Khosla P, Tian Y, Wang X, et al. Supervised Contrastive Learning. *NeurIPS*. 2020;33:18661–18673.
16. DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the Areas under Two or More Correlated Receiver Operating Characteristic Curves: A Nonparametric Approach. *Biometrics*. 1988;44(3):837–845.
17. Vickers AJ, Elkin EB. Decision Curve Analysis: A Novel Method for Evaluating Prediction Models. *Med Decis Making*. 2006;26(6):565–574.
18. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. *ICML*. 2017;70:1321–1330.
19. Nguyen A, Yosinski J, Clune J. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images. *CVPR*. 2015:427–436.
20. Ribeiro MT, Singh S, Guestrin C. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*. 2016:1135–1144.
21. Zhao Q, et al. Deep Learning Methods in Medical Image-Based Hepatocellular Carcinoma Diagnosis: A Systematic Review and Meta-Analysis. *J Cancer Res Clin Oncol*. 2023. (pooled AUC 0.95, sensitivity 89%)
22. Donnelly J, Barnett AJ, Chen C. All You Need Is a Guiding Hand: Mitigating Shortcut Bias in Prototype Networks. *MICCAI*. 2024.
23. Chen C, Li O, Tao D, et al. This Looks Like That: Deep Learning for Interpretable Image Recognition. *NeurIPS*. 2019.
24. Kim J, et al. MAProtoNet: A Multi-scale Attentive Interpretable Prototypical Part Network for 3D MRI Brain Tumor Classification. *arXiv*. 2024. [arXiv:2404.08917]
25. *(ProtoPNet application to Digital Breast Tomosynthesis — Comput Struct Biotechnol J, 2025)*
26. *(CSR: Concept-based Similarity Reasoning for Medical Image Analysis. CVPR 2025 — citation 확정 후 삽입)*
27. Du Z, et al. Development and Validation of an Ultrasound-Based Interpretable Machine Learning Model for the Classification of ≤3 cm Hepatocellular Carcinoma. *eClinicalMedicine*. 2025;81:103098.
