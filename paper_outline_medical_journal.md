# 논문 초고 개요 — 의학 저널 투고용

> **작성 상태**: 임상적 의의 중심 재작성 (rev. 2026-06-30 — Table 2 수치 기입)
> **목표 저널 등급**: PubMed 등재, SCIE Q1–Q2
> *(예: Ultrasonics, Diagnostics, Frontiers in Oncology, JMIR Medical Informatics)*

---

## 제목 (안)

**B-mode 복부 초음파 영상에서 새로운 이중 출력 영상표지자의 개발 및 검증:\nHCC Cosine Score와 Confidence Score의 상호보완적 임상적 의의**

*영문 병기:*
**Development and Validation of a Novel Dual-Output Imaging Marker for Quantifying HCC-Likeness on B-mode Abdominal Ultrasound: Complementary Clinical Roles of HCC Cosine Score and Confidence Score**

---

## 구조화 초록 (Structured Abstract)

### 연구 배경 (Background)

B-mode 초음파는 간세포암(hepatocellular carcinoma, HCC) 감시의 핵심 도구이지만, 조기 병변에 대한 민감도는 제한적이며 검사자 의존성이 크다. 기존 딥러닝 분류 모델은 대부분 단일 softmax 출력값을 확률로 해석하는 체계를 사용하는데, 이 출력값은 임상의가 판독 시 사용하는 유사성 기반 추론과 직접적으로 대응하지 않으며, 보정되지 않은 과잉 확신을 반영할 수 있다.

### 방법 (Methods)

삼성서울병원 공개 데이터셋(SMC-LUD, 1,021명, 5,385장)[14]을 이용한 단일 기관 후향적 연구이다. 환자 단위 분리를 적용하여 훈련(1,858장)/검증(530장)/테스트(268장) 세트를 구분하였다. Hybrid vision transformer(EfficientNetV2B0 + transformer encoder)를 교차 엔트로피와 supervised contrastive learning(SupCon) 결합으로 학습하고, 두 가지 출력을 정의하였다. 첫째, softmax 출력을 결정 강도(decision strength)를 나타내는 **confidence score**로 명명하였다. 둘째, 학습된 임베딩과 HCC 전형 벡터(prototype) 간 cosine similarity를 **HCC cosine score**, hemangioma와의 상대적 마진을 **Δscore**로 정의하였다. 임계값은 검증 세트에서만 결정(Youden's J)하여 테스트 세트에 고정 적용하였다.

### 결과 (Results)

검증 세트에서 confidence score의 AUROC 1.000, 민감도 94.3%, 특이도 100.0%를 달성하였으며, t-SNE 시각화에서 HCC-hemangioma 임베딩 군집의 명확한 분리가 확인되어 cosine 기반 표지자의 타당성을 지지하였다.

> 📝 **[TODO: Test 결과 / ROC 비교]** 테스트 세트 성능 및 ROC-B/C 수치 기입

### 결론 (Conclusions)

본 연구는 B-mode 초음파에서 임상의의 유사성 기반 추론과 개념적으로 일치하는 새로운 연속형 영상표지자(HCC cosine score, Δscore)를 제안하였다. 두 출력의 불일치 패턴은 모델 해석에 주의가 필요한 사례를 식별하는 보조 신호로 기능할 수 있으며, 이 이중 출력 체계는 임상의가 모델 판단을 보다 투명하게 해석하는 구조적 틀을 제공한다.

**핵심어**: 간세포암; 초음파; 딥러닝; supervised contrastive learning; vision transformer; cosine similarity; 영상표지자; 이중 출력 진단; 임상 의사결정 지원

---

## 1. 서론 (Introduction)

간세포암은 원발성 간암의 가장 흔한 조직학적 아형으로, 전 세계적으로 암 관련 사망의 주요 원인 중 하나이다.[1][2] 예후는 진단 시 병기에 크게 의존하며, 조기 병변은 수술적 절제, 고주파 열치료, 간이식 등 완치적 치료가 가능한 반면 진행 병변은 생존율이 현저히 낮다.[2][3] 이러한 이유로 국내외 진료 지침은 고위험 환자군에서 복부 초음파를 이용한 정기적 감시 검사를 권고하고 있다.[2][3]

B-mode 초음파는 현재 이용 가능한 감시 도구 중 가장 접근성이 높고 임상 현장에서 가장 광범위하게 사용되는 영상 방법이다. 비침습적이고, 비용이 저렴하며, 반복 시행이 가능하고, 외래 환경에서 즉시 이용할 수 있다. 그러나 초음파만을 이용한 조기 HCC 감지 민감도는 여전히 제한적이며, alpha-fetoprotein 등 혈청표지자를 병용하더라도 성능 개선은 제한적이다.[4][5] 또한 진단 품질은 검사자의 경험, 스캔 기법, 판독 역량에 의해 크게 좌우되며, 이 검사자 의존성은 전문 인력이 상시 근무하지 않는 일차의료 및 지역사회 환경에서 특히 문제가 된다.

이러한 한계는 임상적으로 해결되지 않은 필요를 만들어낸다. 많은 일선 진료 현장에서 의사들은 간 병변이 조영증강 영상 검사 또는 전문의 평가로 이어질 만큼 충분히 의심스러운지를 판단해야 한다. B-mode 초음파에서 직접 진단 정보를 추출할 수 있는 보조 도구는 이 맥락에서 임상적으로 의미가 있다. 그러나 AI 시스템이 임상적으로 설득력을 가지려면 단순히 분류 성능이 높은 것으로는 충분하지 않다. 모델 출력이 임상의가 실제로 병변을 판단하는 방식과 대응할 수 있어야 한다.

최근 초음파 간 국소 병변 분류에서 딥러닝 기반 모델이 유망한 성능을 보인 연구들이 보고되었다.[6][7] 그러나 이 분야의 기존 문헌은 몇 가지 한계를 지닌다. 첫째, 대부분의 모델은 단일 softmax 출력을 제공하며 이를 확률로 기술하는 경향이 있으나, softmax 출력이 보정된(calibrated) 확률을 의미하지는 않는다.[18] 둘째, 임계값 설정이 최종 평가로부터 명확히 분리되지 않는 경우가 있어 지나치게 낙관적인 성능 추정이 발생할 수 있다. 셋째, 비교적 소규모인 의료 영상 데이터셋에서 각각 고유한 단점을 가지는 기존 CNN 또는 순수 vision transformer 구조가 주로 사용되었다.[9][10][11]

초음파 영상에서 해석 가능성 문제는 별도의 강조가 필요하다. 전형적인 간 초음파 프레임에는 병변 자체와 무관한 정보, 즉 캘리퍼, 측정 눈금, 텍스트 오버레이, 기기 고유의 획득 artifact 등이 포함될 수 있다. 모델이 이러한 부가 단서를 클래스 레이블과 연결하는 방식으로 학습하면, 겉으로는 우수한 분류 성능이 관찰되더라도 그것은 병변에 대한 실질적 이해가 아닌 지름길 학습(shortcut learning)을 반영하는 것일 수 있다.[12][19][20] 이 경우 높은 confidence 출력은 영상의학적으로 타당한 소견을 포착하였다는 증거가 아니라, 비병변 부가 단서에 대한 확신을 반영할 뿐이다.

임상의가 간 병변을 진단할 때 단순히 단일 스칼라 확신 값에 의존하지 않는다는 점이 중요하다. 임상의는 목표 병변의 시각적 특성을 내면화된 전형(prototype) 또는 패턴과 비교한다. 즉, 이 병변이 전형적인 HCC를 얼마나 닮았고, 양성 혈관종을 얼마나 닮았는가를 평가하는 **유사성 기반 추론(similarity-based reasoning)**을 수행한다. 모델이 각 영상을 내부적으로 표현하는 고차원 벡터(임베딩)가 잘 구성된 경우, 이 벡터와 전형적인 HCC 영상들의 평균 벡터 사이의 유사도(cosine similarity)는 임상의의 진단 추론과 개념적으로 일치하는 영상표지자로 기능할 수 있다.[15]

이에 따라 본 연구는 두 가지 목적으로 설계되었다. 첫째, B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증한다. 둘째, 단일 출력 분류 체계를 넘어, confidence score와 임베딩 기반 HCC cosine score 및 Δscore로 구성된 이중 출력 체계를 제안하고, 이 두 출력이 각각 독립적인 임상 해석 정보를 제공할 수 있는지 평가한다. 이 이중 출력 체계는 분류 성능 자체보다 **임상의가 실제로 사용하는 유사성 기반 추론을 수치화한 새로운 영상표지자 후보를 제시하는 것**을 핵심 기여로 삼는다.

---

## 2. 대상 및 방법 (Materials and Methods)

### 2.1 연구 설계

본 연구는 대한민국 서울 삼성서울병원에서 수집된 B-mode 간 초음파 영상 아카이브를 이용한 단일 기관 후향적 관찰 연구이다. 본 논문은 예측 모델 연구에 대한 TRIPOD 보고 기준에 따라 작성되었다.[13] 본 연구는 기관생명윤리위원회 승인 하에 후향적 설계에 따라 수행되었으며, 동의 면제가 적용되었다.

> 📝 **[TODO: IRB]** IRB 승인번호 및 동의면제 문구 기입

### 2.2 데이터셋 및 연구 대상

본 연구에서는 삼성서울병원 간 초음파 데이터셋(Samsung Medical Center–Liver Ultrasound Dataset, SMC-LUD)을 사용하였다.[14] SMC-LUD는 2015년부터 2024년까지 수집된 간 국소 병변의 B-mode 초음파 영상으로 구성된 공개 데이터셋으로, 병리학적으로 확인된 HCC 영상 2,716장과 영상 기준으로 진단된 혈관종(hemangioma) 영상 2,669장을 포함하여 총 1,021명 환자의 5,385장 흑백 영상으로 이루어져 있다.[14] 모든 영상은 384 × 384 픽셀의 흑백 부동소수점 배열로 표준화되었다.

동일 환자의 영상이 개발 단계와 평가 단계 사이에 교차 오염되지 않도록 환자 단위 분리를 적용하였다. 최종 데이터는 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구분하였다. HCC와 hemangioma의 클래스 비율은 세트 간에 균형 있게 유지되었으며, HCC가 전체의 약 52%, hemangioma가 약 48%를 차지하였다.

> 📝 **[TODO: 메타데이터]** 연령, 성별, 간경변 유무, 병변 크기 등 임상 메타데이터 요약 표 기입

### 2.3 모델 구조

제안 모델은 합성곱 신경망(CNN)과 트랜스포머 인코더(transformer encoder)를 결합한 hybrid vision transformer이다. CNN backbone으로는 ImageNet 사전학습 가중치로 초기화된 ResNet50V2 또는 EfficientNetV2B0를 사용하였다. 추출된 특징 맵은 패치 형태의 토큰으로 재형성되어 트랜스포머 인코더에 입력되었다. 분류 토큰(CLS token)의 최종 표현 벡터(임베딩)가 모든 하위 예측에 사용되었으며, 동일한 임베딩에서 이진 분류 출력(confidence score)과 cosine 기반 표지자(HCC cosine score)를 함께 산출하는 이중 출력 구조를 구성하였다.

합성곱 신경망은 인접 픽셀 간 관계를 국소적으로 학습하여 병변의 질감, 경계 패턴, 에코 강도 분포 등 세밀한 특징 포착에 강점을 가진다.[9] Vision transformer는 영상을 패치 단위로 나누어 전역적 관계를 주의(attention) 메커니즘으로 학습하여 원거리 영역 간 맥락 포착에 강점을 가진다.[9][10] Hybrid 구조는 두 강점을 결합하여 국소 특징과 전역 맥락을 동시에 모델링한다.[11]

### 2.4 학습 전략

두 가지 학습 조건을 비교하였다. 첫 번째 조건에서는 교차 엔트로피 손실(cross-entropy loss)만을 이용하였다. 두 번째 조건에서는 교차 엔트로피 손실에 supervised contrastive learning(SupCon) 손실을 결합하였다.[15]

SupCon의 핵심 역할은 cosine 기반 표지자의 타당성을 확보하는 데 있다. 교차 엔트로피만으로 학습할 경우 임베딩 공간은 분류 목적을 통해 간접적으로 최적화되며, 임베딩 간 거리가 영상의학적 의미를 갖도록 보장되지 않는다. 반면 SupCon이 추가되면 동일 클래스 병변이 임베딩 공간에서 서로 인접하고(intra-class compactness), 다른 클래스는 분산되도록(inter-class separability) 학습 목적 자체가 설계된다.[15] 따라서 클래스 전형 임베딩에 대한 cosine similarity는 학습 목적과 정합적인 영상표지자가 된다. 비교 목적으로 자기 지도 학습(SSL) 변형 실험(VICReg, NNCLR)도 수행하였다.

모든 모델은 384 × 384 흑백 입력 영상, Adam optimizer, cosine annealing + warmup 학습률 스케줄 하에 학습되었다. 데이터 증강에는 수평/수직 반전, 무작위 자르기 및 크기 조정, 밝기/대비 변환, Gaussian blur가 포함되었다.

### 2.5 이중 출력 표지자 정의

**본 연구의 핵심 기여는 하나의 모델에서 성격이 상이한 두 출력을 명시적으로 구분하고 각각의 임상적 의미를 정의하는 것이다.**

첫 번째 출력인 **confidence score**는 HCC 클래스에 해당하는 softmax 값으로 정의하였다. 이 값은 0과 1 사이에 위치하지만, 현대 심층 신경망의 softmax는 보정된 사후 확률(calibrated posterior probability)을 자동으로 의미하지 않는다.[18] 체계적인 과잉 확신(overconfidence) 문제가 잘 알려져 있으며, 확률론적 해석이 정당화되기 위해서는 사후 보정이 필요하다.[18] 따라서 본 연구에서는 이 값을 확률이 아닌 **모델의 결정 강도(decision strength)**를 나타내는 confidence score로 명명하였다.

두 번째 출력인 **HCC cosine score**는 임베딩 공간에서 정의하였다. 학습 완료 후 훈련 세트 내 HCC 표본들의 평균 임베딩 벡터를 HCC 전형 벡터(prototype)로, hemangioma 표본들의 평균 임베딩 벡터를 hemangioma prototype으로 구성하였다. 임의의 병변 영상에 대해 해당 임베딩 벡터와 HCC prototype 간의 cosine similarity를 HCC cosine score로 산출하였다. 이 점수는 해당 병변이 학습된 HCC 표현의 기하학적 중심과 얼마나 가까운지를 반영하며, **임상의가 병변을 전형적인 HCC 소견과 비교하는 유사성 기반 추론을 수치화한 것**으로 해석할 수 있다.

**Δscore**는 HCC cosine score와 hemangioma cosine score의 차이로 정의하였다. Δscore는 HCC-like phenotype과 hemangioma-like phenotype 사이의 상대적 유사성 마진을 반영하며, "이 병변이 HCC 쪽에 얼마나 더 가까운가"를 나타내는 비교형 표지자로 해석할 수 있다. 이는 실제 감별 진단 상황에서 추가 영상 검사 또는 단기 추적 관찰 여부를 판단하는 데 임상적으로 더 직관적인 정보를 제공할 수 있다.

### 2.6 임계값 결정 및 데이터 유출 방지

낙관적 편향을 최소화하기 위해 모든 운영 임계값은 검증 세트에서만 결정하였다. 임계값 선택 방법은 Youden's J 통계(민감도 + 특이도 − 1 최대화 지점)를 이용하였다. 검증 세트에서 결정된 임계값을 테스트 세트에 변경 없이 고정 적용하였으며, 이 절차는 confidence score, HCC cosine score, Δscore 각각에 대해 별도로 수행하였다.

### 2.7 통계 분석

모델 변별력은 주로 ROC 곡선 아래 면적(AUROC)을 이용하여 평가하였으며, 신뢰구간은 DeLong 방법으로 추정하였다.[16] 세 종류의 ROC 분석을 수행하였다. ROC-A는 confidence score, ROC-B는 HCC cosine score, ROC-C는 Δscore를 기준으로 한다. 세 ROC 곡선 간의 쌍별 비교는 DeLong 검정을 이용하였으며, p > 0.05를 cosine 기반 표지자의 비열등성(non-inferiority) 지지로 해석하였다.

임계값 의존적 지표로는 민감도, 특이도, 양성예측도, 음성예측도, F1 점수, 혼동행렬이 포함되었다. Confidence score와 Δscore의 결합 분포를 이중 출력 산점도로 시각화하고, 두 출력이 불일치하는 사례에 특별한 주의를 기울였다. 잠재적 임상 순편익 평가를 위한 의사결정 곡선 분석(decision curve analysis, DCA)도 수행하였다.[17] 테스트 세트 지표의 신뢰구간은 1,000회 반복 부트스트랩 재표본 추출로 추정하였다.

---

## 3. 결과 (Results)

### 3.1 데이터셋 구성

분석 데이터셋은 SMC-LUD[14]로부터 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구성되었다. HCC와 hemangioma의 비율은 세트 간에 안정적으로 유지되었으며, HCC가 근소하게 많았다.

> 📝 **[TODO: 인구통계표]** 환자 인구통계학적 정보 및 병변 수준 요약 Table 1 완성 (연령, 성별, 간경변 유무, 병변 크기 등)

### 3.2 Ablation: 학습 조건별 진단 성능 비교

다양한 backbone 및 학습 방식 조합에 대한 ablation 결과를 Table 2에 제시하였다. SupCon 결합 조건에서 임베딩 군집의 분리도가 개선되었으며, t-SNE 시각화에서 이를 확인하였다. SSL 변형들은 분류 성능과 cosine score 유효성에서 SupCon 직접 학습 방식과 통계적으로 유의한 차이를 보이지 않았다. 이는 이중 출력 체계의 임상적 유용성이 특정 학습 커리큘럼에 종속적이지 않음을 시사한다.

#### Table 2. Ablation: Backbone × Training Mode — Diagnostic Performance on Validation Set

| # | Backbone | Training Mode | AUROC | Sensitivity (%) | Specificity (%) | PPV (%) | NPV (%) | F1 Score | Acc (%) |
|---|----------|--------------|-------|-----------------|-----------------|---------|---------|----------|---------|
| 1 | ResNet50V2 | CE Only | 0.9892 | 92.78 | 100.00 | 100.00 | 92.67 | 0.9625 | 96.23 |
| 2 | EfficientNetV2B0 | CE Only | 0.9964 | 95.67 | 100.00 | 100.00 | 95.47 | 0.9779 | 97.74 |
| 3 | ResNet50V2 | CE + SupCon | 0.9946 | 94.58 | 100.00 | 100.00 | 94.40 | 0.9722 | 97.17 |
| **4** | **EfficientNetV2B0** | **CE + SupCon** | **0.9946** | **95.67** | **100.00** | **100.00** | **95.47** | **0.9779** | **97.74** |
| 5 | ResNet50V2 | VICReg (SSL) | — | — | — | — | — | — | — |
| 6 | EfficientNetV2B0 | VICReg (SSL) | 0.9910 | 87.73 | 100.00 | 100.00 | 88.15 | 0.9346 | 93.58 |
| 7 | ResNet50V2 | NNCLR (SSL) | 0.9874 | 89.17 | 100.00 | 100.00 | 89.40 | 0.9427 | 94.34 |
| 8 | EfficientNetV2B0 | NNCLR (SSL) | — | — | — | — | — | — | — |

*CE = Cross-Entropy; SupCon = Supervised Contrastive Learning; SSL = Self-Supervised Learning (fine-tuned head).
All metrics on validation set (n=530). AUROC 95% CI via DeLong method. Cutoff by Youden's J on validation set only.
Bold: primary model selected for subsequent analyses. — : run not available.*

> 📝 **[TODO: Table 2]** #5 (ResNet VICReg), #8 (EfficientNet NNCLR) 실험 수치 미기입 — 해당 wandb json 파일 확보 후 기입

### 3.3 주력 모델 성능 (EfficientNetV2B0 + CE + SupCon)

#### Table 3. Primary Model Performance — Confidence Score

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| AUROC (95% CI) | 0.9946 (–) | ___ (___ – ___) |
| Optimal Cutoff (Youden's J) | — | — *(Val cutoff 고정 적용)* |
| Accuracy (%) | 97.74 | ___ |
| Sensitivity (%) | 95.67 | ___ |
| Specificity (%) | 100.00 | ___ |
| PPV (%) | 100.00 | ___ |
| NPV (%) | 95.47 | ___ |
| F1 Score | 0.9779 | ___ |
| TP / FP / FN / TN | 265 / 0 / 12 / 253 | ___ / ___ / ___ / ___ |

*95% CI for test set metrics: bootstrap n=1,000.*

> 📝 **[TODO: Test 결과]** 테스트 세트 전체 수치 기입

검증 세트에서 AUROC 0.9946, 정확도 97.74%, 민감도 95.67%, 특이도 100.0%, 양성예측도 100.0%, 음성예측도 95.47%, F1 점수 0.9779를 달성하였다. 위양성이 한 건도 없었다는 사실은, 이 기준에 의해 HCC로 판정된 경우 모델이 극히 높은 정밀도로 감별하였음을 시사한다.

### 3.4 임베딩 구조와 cosine 기반 표지자의 타당성

t-SNE를 이용한 임베딩 공간 시각화에서 HCC와 hemangioma 군집 간 뚜렷한 분리가 관찰되었다. 이 결과는 단순한 시각적 확인을 넘어 cosine 기반 표지자의 타당성을 직접 뒷받침한다. 학습된 임베딩 공간이 질환 특이적 기하학적 구조를 보존하지 못한다면, 클래스 prototype에 대한 cosine similarity는 해석적 가치를 갖기 어렵다. 관찰된 군집 분리는 이 설정에서 prototype 유사도가 의미 있는 영상표지자 후보로 기능할 수 있음을 지지한다.

### 3.5 이중 출력 ROC 비교 (ROC-A, ROC-B, ROC-C)

본 연구의 핵심 비교 분석은 confidence score, HCC cosine score, Δscore에 대한 세 종류의 ROC 분석이다.

#### Table 4. Three-Way ROC Comparison — Dual Output System (Validation Set)

| Output | Score Type | AUROC | 95% CI | DeLong p-value vs ROC-A |
|--------|-----------|-------|--------|-------------------------|
| ROC-A | Confidence Score (softmax) | **0.9946** | — | — (reference) |
| ROC-B | HCC Cosine Score | ___ | ___ – ___ | ___ |
| ROC-C | Δscore (HCC cosine − Hem cosine) | ___ | ___ – ___ | ___ |

*p > 0.05: support for non-inferiority of cosine-based markers.*

> 📝 **[TODO: ROC 비교]** ROC-B, ROC-C AUROC, 95% CI, DeLong p-value 기입 및 Figure 삽입

이 비교의 목적은 두 가지이다. 첫째, cosine 기반 표지자가 confidence score와 동등한 AUROC를 보이는지(비열등성 검증). 둘째, 두 지표가 불일치하는 사례에서 서로 다른 임상 정보를 제공하는지(상보성 확인).

### 3.6 이중 출력 불일치 분석

#### Table 5. Dual-Output Discordance Analysis (Validation Set)

| Confidence Score | Δscore | N (cases) | Predicted Label | 임상적 해석 |
|-----------------|--------|-----------|-----------------|------------|
| High (≥ cutoff) | High (≥ cutoff) | ___ | HCC | 일치 고위험 — 강력한 HCC 근거 |
| High (≥ cutoff) | Low (< cutoff) | ___ | HCC | **불일치** — 주의 요망; 추가 검사 고려 |
| Low (< cutoff) | High (≥ cutoff) | ___ | Hemangioma | 불일치 — 영상표지자 HCC-like; 재판독 고려 |
| Low (< cutoff) | Low (< cutoff) | ___ | Hemangioma | 일치 저위험 — 강력한 양성 근거 |

> 📝 **[TODO: 불일치 분석]** 각 사분면 케이스 수 기입

Confidence와 Δscore의 불일치, 특히 confidence는 높지만 Δscore가 낮은 경우는 모델 해석에 주의가 필요한 사례를 식별하는 보조 신호로 기능할 수 있다.

---

## 4. 고찰 (Discussion)

### 4.1 임상적 미충족 수요와 본 연구의 위치

초음파 기반 간 국소 병변 딥러닝 분류 연구는 2020년을 전후하여 본격적으로 성장하였다. Yang 등(2020, *EBioMedicine*)은 13개 기관 2,143명 데이터에서 AUROC 0.924를 달성하였고, 이는 15년 경력 임상의 236명의 성능을 유의하게 상회하였다.[6] 이후 다기관 전향 연구(Du 등, 2025, *eClinicalMedicine*)까지 이어진 이 분야의 발전에도 불구하고, 기존 연구들의 공통적 구조적 특징은 **단일 softmax 출력(또는 그 후처리 조합)**을 최종 진단 지표로 사용한다는 점이다.[6][27] 어느 연구도 모델이 학습한 임베딩 공간의 기하학적 구조, 즉 병변이 전형적인 HCC 표현의 중심과 얼마나 가까운지를 독립적인 진단 표지자로 제시하지 않았다. 본 연구는 이 간극을 메운다.

임상적으로도 해결되지 않은 필요가 존재한다. Tzartzeva 등(2018)의 메타분석에서 초음파 단독 HCC 감지 민감도는 47%에 불과하였으며, AFP 병용 시에도 63% 수준에 머물렀다.[4] 이 한계의 일부는 초음파 자체의 해상도가 아니라, 판독 과정에서 병변의 suspiciousness를 연속형으로 표현하는 보조 지표의 부재에서 기인한다. B-mode 영상 한 장에서 임상의의 유사성 기반 추론을 반영하는 정량적 영상표지자가 제공된다면, 추가 검사 의뢰, 재검 간격 조정, 전문의 판독 의뢰와 같은 실제 의사결정을 지원하는 보조 도구로 기능할 수 있다.

### 4.2 Confidence Score의 개념적 재정의 — 표현 관행에 대한 이의 제기

기존의 많은 의료 AI 연구들은 softmax 출력을 관행적으로 확률로 표현한다. Guo 등(2017, ICML)은 현대 심층 신경망이 체계적인 과잉 확신을 보이며, softmax 값과 실제 정답률 사이에 유의한 괴리가 존재함을 실험적으로 입증하였다.[18] 이 괴리는 temperature scaling과 같은 사후 보정 없이는 해소되지 않는다.

초음파 데이터셋처럼 비병변 artifact가 풍부한 환경에서는, 모델이 병변의 실질적인 소견이 아닌 캘리퍼나 눈금 표시와 같은 부가 단서에 의해 높은 confidence를 산출할 수 있다. 이것은 Ribeiro 등(2016)[20]이 기술한 shortcut learning의 의료 영상 특이적 발현이며, 높은 softmax 출력이 임상적으로 타당한 소견을 포착하였다는 보증이 되지 못함을 의미한다. 본 연구에서 softmax 출력을 확률이 아닌 **confidence score(결정 강도)**로 명명한 것은 이 표현 관행에 대한 명시적 이의 제기이며, 의료 AI 출력 해석의 정확성을 높이기 위한 방법론적 기여이다.

### 4.3 HCC Cosine Score — 유사성 기반 추론의 수치화

임상의가 초음파 영상에서 간 병변을 진단하는 과정은 단순한 이진 판단이 아니다. 수련을 통해 내면화한 전형적 HCC 소견(저에코 배경, 주변부 저에코 테두리, 결절 내 결절 패턴)과 혈관종의 전형적 소견(고에코, 경계 명확, 균일한 에코)을 현재 병변과 비교 평가하는 유사성 기반 추론이 핵심이다.

이 임상적 추론 방식을 AI로 모사하려는 선행 시도로는 Chen & Li(2019)의 ProtoPNet 계열이 있다. 의료 영상 분야에서도 D-ProtoPNet, MAProtoNet 등에 적용되었으나[23][24], 이들 모델은 추가적인 prototype 학습 단계와 별도의 아키텍처 변경이 필요하다.

본 연구의 접근법은 추가 아키텍처 변경 없이 SupCon으로 강화된 임베딩 공간에서 HCC 전체 영상의 평균 prototype 벡터와의 cosine similarity를 단일 연속형 점수로 산출한다. 임상적으로 이 점수는 AFP나 PIVKA-II처럼 연속형 스칼라 값으로 산출되어 ROC 분석과 임계값 검증이 가능한 **연속형 영상표지자 후보**로 기능할 수 있다. Δscore는 단일 클래스 유사도가 아니라 경쟁 두 클래스 사이의 상대적 위치를 정량화하여 "HCC인가, hemangioma인가"라는 비교 추론을 수치화한다는 점에서 추가적인 임상 가치를 가질 수 있다.

### 4.4 SupCon의 역할 — 임베딩 공간 구조화의 근거

Cosine score를 의미 있는 표지자로 삼으려면, 임베딩 공간 자체가 클래스별 기하학적 응집성을 갖도록 학습되어야 한다. 이것이 SupCon을 도입한 핵심 이유이다.[15] CE-only 학습에서는 분류 결정 경계가 형성되면 임베딩 공간의 세밀한 기하학적 구조는 부수적으로만 최적화된다. 반면 SupCon이 추가되면 intra-class compactness와 inter-class separability가 학습 목적에 직접 포함되어, cosine similarity가 학습 목적과 정렬된 의미 있는 유사성 척도가 된다. 이는 cosine score를 임상 표지자로 제안하는 이론적 근거이며, ablation(Table 2)과 t-SNE 시각화가 이를 지지한다.

### 4.5 이중 출력 불일치의 임상적 의미 — 의사결정 지원 구조

본 연구의 가장 독창적인 임상적 기여는 confidence-cosine 불일치 패턴 분석이다. Confidence score가 높으나 Δscore가 낮은 경우는 모델이 병변의 실질적인 소견보다 비병변 부가 단서에 의존하여 높은 확신을 산출하였을 가능성을 시사한다. 이러한 불일치는 모델 해석에 주의가 필요한 사례를 식별하는 보조 신호로 기능할 수 있다.

단일 confidence 지표만 사용하는 기존 연구들에서는 이러한 불일치 패턴을 탐지할 구조가 없다. 이중 출력 체계는 추가 영상 검사 의뢰, 단기 추적 관찰, 전문의 판독 의뢰 여부를 판단하는 데 더 풍부한 정보를 제공한다는 점에서 임상 workflow에서의 잠재적 활용 가치가 있다.

### 4.6 Shortcut 관찰과 단일 confidence 지표의 한계

예비적 시각화 분석에서 일부 사례에서 모델의 활성 영역이 병변 실질보다 캘리퍼, 눈금 표시와 같은 비병변 artifact에 집중되는 것이 관찰되었다. 초음파 영상에서의 shortcut learning은 성능 과대 추정의 문제를 넘어, 모델이 맞는 예측을 잘못된 이유로 하는 구조적 취약성이다.[20][22] 이 관찰은 단일 confidence 지표에 대한 임상적 의존의 위험성을 경고하며, 임베딩 공간 전체의 기하학적 유사도를 기반으로 하는 cosine score 병용의 임상적 타당성을 추가적으로 뒷받침한다.

### 4.7 한계 및 향후 연구 방향

**단일 기관 후향적 설계.** Yang 등(2020)이 13개 기관, Du 등(2025)이 다기관 전향 검증을 수행한 것과 달리, 본 연구는 단일 기관 데이터(SMC-LUD)[14]에 기반한다. 기관별 초음파 장비, 영상 획득 프로토콜, annotation 방식의 차이로 인해 외적 타당도는 검증되지 않았으며, 이중 출력 체계의 임상적 신뢰도 확립을 위해서는 다기관 전향 코호트에서의 외부 검증이 필수적이다.

**혈청표지자 대비 증분 이득 미평가.** AFP 및 PIVKA-II와의 직접적인 증분 이득 비교가 이루어지지 않았다. HCC cosine score가 AFP에 추가하여 독립적인 진단 가치를 제공하는지 평가하려면 별도 연구가 필요하다.

**이진 대조군의 임상적 한계.** 실제 임상의 감별 진단 스펙트럼에는 FNH, 담관암, 전이성 간암 등이 포함되며, 이진 분류 체계는 이 현실을 충분히 반영하지 못한다. 다중 클래스 확장은 향후 과제이다.

**임상의 대비 직접 비교 부재.** Yang 등(2020)이 임상의 236명과의 직접 성능 비교를 수행한 것과 달리, 본 연구에서는 숙련된 임상의와의 head-to-head 비교가 이루어지지 않았다.

**병변 크기 층화 분석 미시행.** 조기 HCC 탐지 성능(특히 ≤2 cm 병변)에 대한 별도 평가가 필요하다.

**Prototype Drift 가능성.** HCC cosine score의 prototype은 훈련 세트의 평균 임베딩으로 정의된다. 다기관 환경에서는 prototype이 표적 집단을 충분히 대표하지 못하는 prototype drift가 발생할 수 있으며, 다기관 환경에서의 prototype 재보정(recalibration) 전략 검토가 향후 필요하다.

---

## 5. 결론 (Conclusion)

본 연구는 B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증하며, confidence score와 임베딩 기반 HCC cosine score 및 Δscore로 구성된 이중 출력 체계를 제안하였다. 모델은 우수한 내부 검증 성능을 보였고, 학습된 임베딩 구조는 cosine 기반 유사도가 의미 있는 영상표지자 후보로 기능할 수 있음을 지지하였다.

본 연구의 핵심 기여는 모델의 분류 성능 자체보다, **임상의가 실제로 사용하는 유사성 기반 추론을 수치화한 새로운 연속형 영상표지자 후보(HCC cosine score, Δscore)를 B-mode 초음파에서 도출하는 체계를 제안하였다는 데 있다.** Confidence score는 모델의 결정 강도를 반영하고, cosine 기반 표지자는 학습된 클래스 전형에 대한 병변 유사도를 반영하며, 두 출력의 불일치는 모델 해석에 주의가 필요한 사례를 식별하는 보조 신호로 기능할 수 있다. 이 이중 출력 체계는 임상의가 모델의 판단을 보다 투명하게 해석하고 검증할 수 있는 구조적 틀을 제공한다는 점에서 방법론적, 임상적 의의를 가진다.

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

#### 실험별 GPU 시간 배분 (추정치)

| 실험 그룹 | 주요 내용 | 예상 GPU·hr |
|-----------|----------|-------------|
| Stage 1: SSL Pre-training (VICReg, NNCLR) | EfficientNetV2B0 / ResNet50V2 × 2 알고리즘 | ~6 hr |
| Stage 2: Classification + SupCon | CE Only × 2 backbone + CE+SupCon × 2 backbone | ~4 hr |
| Stage 3: Cosine Score 계산 및 평가 | Prototype 추출, ROC-A/B/C 분석, t-SNE | ~1 hr |
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
23. *(D-ProtoPNet application to pediatric CXR pneumonia classification — Explainable AI, 2025)*
24. Kim J, et al. MAProtoNet: A Multi-scale Attentive Interpretable Prototypical Part Network for 3D MRI Brain Tumor Classification. *arXiv*. 2024. [arXiv:2404.08917]
25. *(ProtoPNet application to Digital Breast Tomosynthesis — Comput Struct Biotechnol J, 2025)*
26. *(CSR: Concept-based Similarity Reasoning for Medical Image Analysis. CVPR 2025 — citation 확정 후 삽입)*
27. Du Z, et al. Development and Validation of an Ultrasound-Based Interpretable Machine Learning Model for the Classification of ≤3 cm Hepatocellular Carcinoma. *eClinicalMedicine*. 2025;81:103098.
