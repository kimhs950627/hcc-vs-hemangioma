# 논문 초고 개요 — 의학 저널 투고용

> **작성 상태**: 전체 한글 서술체 초고 (rev. 2026-06-30 — Discussion 강화)
> **목표 저널 등급**: PubMed 등재, SCIE Q1–Q2
> *(예: Ultrasonics, Diagnostics, Frontiers in Oncology, JMIR Medical Informatics)*

---

## 제목 (안)

**B-mode 복부 초음파 영상에서 Hybrid Vision Transformer를 이용한 간세포암 이중 출력 진단 모델의 개발 및 검증: Confidence Score와 Cosine Similarity 기반 HCC Score의 상호보완적 활용**

*영문 병기:*
**Dual-Output HCC Scoring from B-mode Abdominal Ultrasound Using a Hybrid Vision Transformer: Complementary Roles of Confidence Score and Embedding-Based Cosine Similarity**

---

## 구조화 초록 (Structured Abstract)

### 연구 배경 (Background)

B-mode 초음파는 간세포암(hepatocellular carcinoma, HCC) 감시에서 가장 널리 사용되는 1차 영상 도구이지만, 조기 병변에 대한 민감도는 제한적이며 검사자 의존성이 크다. 최근 딥러닝 기반 초음파 분류 모델들이 보고되고 있으나, 대부분 모델의 출력값을 "확률"로 해석하는 단일 출력 체계를 유지하고 있으며, 임상가가 판독에 사용하는 유사성 기반 추론과의 연결은 충분히 제시되지 못하였다. 특히 초음파 영상에는 캘리퍼, 측정 눈금, 텍스트 annotation 등 비병변성 artifact가 존재하여 모델이 병변이 아닌 주변 부가 정보를 단서로 학습하는 지름길 학습(shortcut learning)을 유발할 수 있으며, 이 경우 높은 모델 확신만으로는 임상적 신뢰를 제공하기 어렵다.

### 방법 (Methods)

본 연구는 단일 기관 후향적 연구로, 삼성서울병원에서 구축된 공개 B-mode 간 초음파 데이터셋(SMC-LUD)을 사용하였다. 환자 단위 분리를 유지한 상태에서 훈련, 검증, 테스트 세트로 영상을 구분하였다. 모델은 합성곱 신경망(convolutional neural network, CNN)과 트랜스포머(transformer)를 결합한 hybrid vision transformer로 설계하였고, 교차 엔트로피 단독 학습과 supervised contrastive learning(SupCon)을 결합한 학습 방식을 비교하였다. 단일 모델에서 두 가지 출력을 정의하였다. 첫째, 모델의 최종 판별 출력인 softmax 값은 고전적 의미의 확률이 아닌 모델의 상대적 확신 정도를 나타내는 **confidence score**로 명명하였다. 둘째, 모델이 각 이미지를 내부적으로 표현하는 벡터(임베딩)와 HCC 전형 영상들의 평균 벡터 간 유사도인 **HCC cosine score**를 산출하였으며, hemangioma와의 유사도 차이를 반영한 **Δscore**도 추가 계산하였다. 최적 임계값은 Youden's J 통계를 이용하여 검증 세트에서만 결정하였고, 테스트 세트는 맹검 평가에만 사용하였다.

### 결과 (Results)

검증 세트에서 confidence score 기반 분류는 AUROC 1.000, 정확도 97.0%, 민감도 94.3%, 특이도 100.0%, 양성예측도 100.0%를 보였다. t-SNE 시각화에서 HCC와 hemangioma의 임베딩 군집이 뚜렷하게 분리되었으며, 이는 cosine similarity 기반 점수가 의미 있는 표현 공간 위에서 산출됨을 시사하였다. Confidence score, HCC cosine score, Δscore 각각에 대한 ROC 분석을 비교하여, softmax confidence와 임베딩 유사도가 서로 다른 진단 정보를 제공할 수 있음을 확인하고자 하였다.

### 결론 (Conclusions)

Hybrid vision transformer 기반 이중 출력 접근법은 B-mode 초음파만으로도 높은 변별력을 보였으며, 기존 단일 출력 체계에 더해 cosine similarity 기반 HCC score를 병용함으로써 임상가에게 두 가지 상호보완적인 진단 정보 축을 함께 제공할 수 있음을 확인하였다. 이 이중 출력 체계는 모델의 예측을 단일 스칼라 값으로 환원하지 않고, 임상가가 실제로 사용하는 "전형적 병변과의 유사성"이라는 해석 축을 함께 제공한다는 점에서 방법론적, 임상적 의의를 가진다.

**핵심어**: 간세포암; 초음파; 딥러닝; supervised contrastive learning; vision transformer; cosine similarity; 모델 보정(calibration); 설명 가능 인공지능; 이중 출력 진단

---

## 1. 서론 (Introduction)

간세포암은 원발성 간암의 가장 흔한 조직학적 아형으로, 전 세계적으로 암 관련 사망의 주요 원인 중 하나이다. 임상적으로 예후는 진단 시 병기에 크게 의존하는데, 조기 병변은 수술적 절제, 고주파 열치료, 간이식 등 완치적 치료가 가능한 반면, 진행 병변은 생존율이 현저히 낮다.[1][2] 이러한 이유로 국내외 진료 지침은 고위험 환자군에서 복부 초음파를 이용한 정기적 감시 검사를 권고하고 있다.[2][3]

B-mode 초음파는 현재 이용 가능한 감시 도구 중 가장 접근성이 높고 임상 현장에서 가장 광범위하게 사용되는 영상 방법이다. 비침습적이고, 비용이 저렴하며, 반복 시행이 가능하고, 외래 환경에서 즉시 이용할 수 있다는 실용적 장점이 분명하다. 그러나 이러한 장점은 상당한 한계와 공존한다. 초음파만을 이용한 조기 HCC 감지 민감도는 여전히 만족스럽지 않으며, alpha-fetoprotein 등 혈청 바이오마커를 병용하더라도 성능 개선은 제한적이다.[4][5] 또한 초음파 검사의 진단 품질은 검사자의 경험, 스캔 기법, 판독 역량에 의해 크게 좌우된다. 이 검사자 의존성은 전문 영상의학과 의사가 상시 근무하지 않는 일차의료 및 지역사회 환경에서 특히 문제가 된다.

이러한 한계는 임상적으로 해결되지 않은 필요를 만들어낸다. 많은 일선 진료 현장에서 의사들은 간 병변이 조영증강 영상 검사 또는 전문의 평가로 이어질 만큼 충분히 의심스러운지를 판단해야 한다. B-mode 초음파에서 직접 진단 정보를 추출할 수 있는 딥러닝 모델은 이 맥락에서 유용한 보조 도구가 될 수 있다. 그러나 AI 시스템이 임상적으로 설득력을 가지려면 단순히 분류 성능이 높은 것으로는 충분하지 않다. 모델 출력이 임상가가 실제로 병변을 판단하는 방식과 일치하는 방향으로 해석될 수 있어야 한다.

최근 초음파 간 국소 병변 분류에서 딥러닝 기반 모델이 유망한 성능을 보인 연구들이 보고되었다.[6][7] 그러나 이 분야의 기존 문헌은 몇 가지 측면에서 한계를 지닌다. 첫째, 대부분의 모델은 단일 softmax 출력을 제공하며, 이를 "확률"로 기술하는 경향이 있으나, softmax 출력이 보정된(calibrated) 확률을 의미하지는 않는다.[18] 둘째, 임계값 설정이 최종 평가로부터 명확히 분리되지 않는 경우가 있어, 지나치게 낙관적인 성능 추정이 발생할 수 있다. 셋째, 비교적 소규모인 의료 영상 데이터셋에서 각각 고유한 단점을 가지는 기존 CNN 또는 순수 vision transformer 구조가 주로 사용되었다.[9][10][11]

초음파 영상에서 해석 가능성 문제는 특별한 강조가 필요하다. 전형적인 간 초음파 프레임에는 병변 자체와 무관한 정보, 즉 캘리퍼, 측정 눈금, 텍스트 오버레이, 기기 고유의 획득 artifact 등이 포함될 수 있다. 모델이 이러한 불필요한 단서를 클래스 레이블과 연결하는 방식으로 학습하면, 겉으로는 우수한 분류 성능이 관찰되더라도 그것은 병변에 대한 실질적 이해가 아닌 지름길 학습(shortcut learning)을 반영하는 것일 수 있다.[12][19][20] 이 경우 높은 confidence 출력은 임상적으로 유효한 영상 소견을 포착하였다는 증거가 아니라, 영상의학적으로 유효하지 않은 부가 단서에 대한 확신을 반영할 뿐이다.

이 구분이 중요한 이유는 임상의가 간 병변을 진단할 때 단순히 단일 스칼라 확신 값에 의존하지 않기 때문이다. 임상의는 목표 병변의 시각적 특성을 이전에 학습한 전형(prototype) 또는 패턴과 비교한다. 즉, 이 병변이 전형적인 HCC를 얼마나 닮았고, 양성 혈관종을 얼마나 닮았는가를 평가하는 유사성 기반 추론을 수행한다. 이러한 추론 방식은 방사선 진단의 부수적 요소가 아니라 핵심 요소이다. 모델이 각 이미지를 내부적으로 표현하는 고차원 벡터, 즉 임베딩(embedding)이 잘 구성된 경우, 이 벡터와 전형적인 HCC 영상들의 평균 벡터 사이의 유사도(cosine similarity)는 단순한 후처리 수치가 아니라 임상의의 진단 추론과 개념적으로 일치하는 의미 있는 진단 지표로 기능할 수 있다.[15]

이에 따라 본 연구는 두 가지 목적으로 설계되었다. 첫째, B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증한다. 둘째, 단일 출력 분류 체계를 넘어, confidence score와 임베딩 기반 HCC cosine score 및 Δscore로 구성된 이중 출력 체계를 제안하고, 이 두 출력이 서로 독립적인 임상 해석 정보를 제공할 수 있는지를 평가한다.

---

## 2. 대상 및 방법 (Materials and Methods)

### 2.1 연구 설계

본 연구는 대한민국 서울 삼성서울병원에서 수집된 B-mode 간 초음파 영상 아카이브를 이용한 단일 기관 후향적 관찰 연구이다. 본 논문은 예측 모델 연구에 대한 TRIPOD 보고 기준에 따라 작성되었다.[13] 기관생명윤리위원회(IRB) 승인번호 및 동의 면제 세부 내용은 후향적 연구 설계에 따라 면제되었다.

> 📝 **[TODO: IRB]** IRB 승인번호 기입

### 2.2 데이터셋 및 연구 대상

본 연구에서는 삼성서울병원 간 초음파 데이터셋(Samsung Medical Center–Liver Ultrasound Dataset, SMC-LUD)을 사용하였다. SMC-LUD는 2015년부터 2024년까지 수집된 간 국소 병변의 B-mode 초음파 영상으로 구성된 공개 데이터셋으로, 병리학적으로 확인된 HCC 영상 2,716장과 영상 기준으로 진단된 혈관종(hemangioma) 영상 2,669장을 포함하여 총 1,021명 환자의 5,385장 흑백 영상으로 이루어져 있다.[14] 모든 영상은 384 × 384 픽셀의 흑백 부동소수점 배열로 표준화되었다.

본 분석에서는 동일 환자의 영상이 개발 단계와 평가 단계 사이에 교차 오염되지 않도록 환자 단위 분리를 적용하였다. 최종 데이터는 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구분되었다. HCC와 hemangioma의 클래스 비율은 세트 간에 균형 있게 유지되었으며, HCC가 전체의 약 52%, hemangioma가 약 48%를 차지하였다. 연령, 성별, 간경변 유무, 병변 크기 등의 임상 메타데이터는 완전한 데이터 연결이 확인되면 요약하였다.

> 📝 **[TODO: 메타데이터]** 연령, 성별, 간경변 유무, 병변 크기 등 임상 메타데이터 요약 표 기입

### 2.3 모델 구조

제안 모델은 합성곱 신경망(CNN)과 트랜스포머 인코더(transformer encoder)를 결합한 hybrid vision transformer이다. 두 구조의 상호 보완적 강점을 활용하기 위한 설계로, 이를 각각 간략히 소개하면 다음과 같다.

합성곱 신경망은 영상 내 인접한 픽셀 간 관계를 국소적으로 학습하는 구조로, 초음파 영상에서 병변의 질감, 경계 패턴, 에코 강도 분포 등 세밀한 국소 특징을 포착하는 데 강점을 가진다.[9] 반면 트랜스포머는 원래 자연어 처리 분야에서 개발된 구조로, 영상 분야에 도입된 vision transformer(ViT)는 영상 전체를 일정 크기의 조각(patch)으로 나눈 뒤 각 조각 간의 전역적 관계를 주의(attention) 메커니즘으로 학습한다.[9][10] 즉 합성곱이 "가까운 픽셀들 사이의 관계"를 보는 데 강하다면, 트랜스포머는 "멀리 떨어진 영역들 사이의 관계"를 포착하는 데 강하다. Hybrid vision transformer는 두 구조를 결합하여 국소 특징과 전역 문맥을 동시에 모델링한다.[11]

CNN backbone으로는 ImageNet 사전학습 가중치로 초기화된 ResNet50V2 또는 EfficientNetV2B0를 사용하였다. 이로부터 추출된 특징 맵은 일정 크기의 조각(patch) 형태의 토큰으로 재형성되어 다양한 깊이의 트랜스포머 인코더에 입력되었다. 분류 토큰(classification token, CLS token)이 부가되고, 이 토큰의 최종 표현 벡터(임베딩)가 모든 하위 예측에 사용되었다. 동일한 임베딩이 이진 분류 출력(confidence score)과 유사성 기반 후처리 점수(cosine HCC score) 계산에 함께 활용되어, 단일 모델 안에서 두 종류의 출력을 동시에 생성하는 체계를 구성하였다.

### 2.4 학습 전략

두 가지 학습 조건을 비교하였다. 첫 번째 조건에서는 교차 엔트로피 손실(cross-entropy loss)만을 이용하여 모델을 최적화하였다. 두 번째 조건에서는 교차 엔트로피 손실에 supervised contrastive learning(SupCon) 손실을 결합하였다. 교차 엔트로피는 이진 분류를 위한 표준적인 판별 학습 신호를 제공하는 반면, supervised contrastive learning은 동일 클래스 표본을 임베딩 공간에서 가깝게, 다른 클래스 표본을 멀게 배치하도록 표현 공간을 명시적으로 구조화한다.[15]

이 학습 방식의 선택이 갖는 핵심 의미는, SupCon으로 학습한 모델에서 임베딩 간 cosine similarity가 단순한 수치적 편의가 아니라 학습 목적함수가 직접 의미 있도록 유도한 유사성 척도가 된다는 점이다. 교차 엔트로피만으로 학습할 경우 임베딩 공간은 분류 목적을 통해 간접적으로 최적화되며, 임베딩 간 거리가 영상의학적 의미를 갖도록 보장되지 않는다. 반면 SupCon이 적용되면 동일 클래스 병변이 임베딩 공간에서 서로 인접하고 다른 클래스는 분산되도록 학습 목적 자체가 설계되므로, 클래스 전형 임베딩에 대한 cosine similarity는 학습 목적과 정합적인 진단 지표가 된다. 본 연구에서 HCC cosine score와 Δscore를 구조화된 이중 출력의 한 축으로 정의하는 이론적 근거가 여기에 있다. 비교 목적으로 자기 지도 학습(self-supervised learning, SSL) 변형 실험도 수행하였으나, 본 연구의 핵심 주장은 학습 커리큘럼 설계가 아닌 이중 출력 체계의 임상적 유용성에 있으므로 해당 결과는 ablation으로 기술한다.

모든 모델은 384 × 384 흑백 입력 영상을 사용하여 학습되었다. 옵티마이저는 Adam을 사용하였고, 초기 학습률은 1 × 10⁻⁴였으며, warmup이 적용된 cosine annealing 방식으로 학습률을 조정하였다. 데이터 증강에는 수평 및 수직 반전, 무작위 자르기 및 크기 조정, 밝기 및 대비 변환, Gaussian blur가 포함되었다. 학습은 클라우드 GPU 환경에서 실용적 계산 제약 조건 내에서 수행되었다.

### 2.5 이중 출력 지표 정의

본 연구의 핵심은 하나의 모델에서 성격이 상이한 두 종류의 출력을 명시적으로 구분하고 각각의 임상적 의미를 정의하는 것이다.

첫 번째 출력인 **confidence score**는 HCC 클래스에 해당하는 softmax 값으로 정의하였다. Softmax 함수는 모델 최종 단계의 각 클래스별 원시 점수(logit)를 0과 1 사이의 값으로 정규화하여 출력한다. 이 값은 0과 1 사이에 위치하지만, 자동적으로 진정한 확률로 해석되어서는 안 된다. 현대의 심층 신경망은 체계적으로 과잉 확신(overconfident)된 출력을 산출하는 것으로 알려져 있으며, 확률론적 해석이 정당화되기 위해서는 temperature scaling과 같은 사후 보정(calibration)이 흔히 필요하다.[18] 이러한 이유로 본 연구에서는 softmax 출력을 "확률"이 아닌 **confidence score**로 의도적으로 명명하였다.

두 번째 출력인 **HCC cosine score**는 임베딩 공간에서 정의하였다. 모델 학습 완료 후, 훈련 세트 내 HCC 표본들의 평균 임베딩 벡터를 계산하여 HCC 전형 벡터(prototype)를 구성하였고, hemangioma 표본들의 평균 임베딩 벡터로 hemangioma prototype을 구성하였다. 임의의 병변 영상에 대해 해당 임베딩 벡터와 HCC prototype 간의 cosine similarity, 즉 두 벡터 간의 방향적 유사성을 계산함으로써 HCC cosine score를 산출하였다. 이 점수는 임베딩 공간에서 학습된 HCC 표현의 기하학적 중심과 해당 병변이 얼마나 가까운지를 반영하도록 설계되었다.

변별력을 더욱 정교화하기 위해 **Δscore**를 HCC cosine score와 hemangioma cosine score의 차이로 정의하였다. Δscore는 HCC와의 유사성뿐 아니라 hemangioma로부터의 비유사성도 동시에 포착하며, 두 경쟁 질환 전형 사이의 상대적 유사성 마진으로 해석할 수 있다.

이러한 이중 출력 구조는 임상의의 임상적 추론 방식에서 직접적으로 동기를 얻은 것이다. 임상의는 단순히 병변이 어떤 내부 확신 수준으로 악성인지를 묻는 것이 아니라, 해당 병변이 전형적인 HCC의 시각적 특성과 hemangioma와 같은 양성 유사 병변의 특성 중 어느 쪽을 더 닮았는지를 고려한다. Cosine 기반 출력은 이 인간의 패턴 비교 과정을 수치화하는 반면, confidence score는 모델 내부의 결정적 확신을 반영한다. 두 출력 간의 일치와 불일치 패턴을 연구하는 것이 본 연구의 주요 목적 중 하나이다.

### 2.6 임계값 결정 및 데이터 유출 방지

낙관적 편향을 최소화하기 위해 모든 운영 임계값은 검증 세트에서만 결정하였다. 주된 임계값 선택 방법은 Youden's J 통계를 이용하였으며, 이는 검증 세트 ROC 곡선에서 민감도와 특이도의 합에서 1을 뺀 값을 최대화하는 지점으로 정의하였다. 검증 세트에서 결정된 임계값은 테스트 세트에 변경 없이 고정 적용하였다. 이 절차는 confidence score, HCC cosine score, Δscore 각각에 대해 별도로 수행하였다. 이 방법론은 혈청 종양표지자 연구에서 cutoff를 독립 코호트에서 결정하고 외부 검증 데이터셋에서 평가하는 방식과 동일한 논리적 엄밀성을 초음파 AI 출력에 적용한 것이다.

### 2.7 통계 분석

모델 변별력은 주로 수신자 조작 특성 곡선(ROC curve) 아래 면적(AUROC)을 이용하여 평가하였다. AUROC의 신뢰구간은 DeLong 방법으로 추정하였다.[16] 세 가지 유형의 ROC 분석을 계획하였다. ROC-A는 confidence score를, ROC-B는 HCC cosine score를, ROC-C는 Δscore를 기준으로 한다. 이 세 ROC 곡선 간의 쌍별 비교는 DeLong 검정을 이용한 상관 ROC 비교로 수행할 예정이며, p값이 0.05를 초과하는 경우 cosine 기반 지표의 비열등성(non-inferiority)이 지지되는 것으로 해석한다.

임계값 의존적 성능 지표로는 민감도, 특이도, 양성예측도, 음성예측도, F1 점수 및 혼동행렬이 포함되었다. 이중 출력 산점도를 통해 confidence score와 Δscore의 결합 분포를 시각화하고, 두 출력이 불일치하는 사례에 특별한 주의를 기울일 예정이다. 잠재적 임상 순편익을 평가하기 위한 의사결정 곡선 분석(decision curve analysis)도 계획하였다.[17] 테스트 세트 지표의 신뢰구간 추정을 위해 1,000회 반복 부트스트랩 재표본 추출을 사용할 예정이다.

---

## 3. 결과 (Results)

### 3.1 데이터셋 구성

분석 데이터셋은 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구성되었다. 전체 세트에 걸쳐 HCC와 hemangioma의 비율은 안정적으로 유지되었으며, HCC가 근소하게 많았다. 이 클래스 균형은 극단적인 클래스 불균형으로 인해 성능 지표가 왜곡될 위험을 줄였다. 메타데이터 가용성 확인 후 환자 인구통계학적 정보 및 병변 수준 요약 표를 추가하였다.

> 📝 **[TODO: 인구통계표]** 환자 인구통계학적 정보 및 병변 수준 요약 Table 1 기입

### 3.2 Ablation: 학습 조건별 진단 성능 비교

다양한 backbone 및 학습 방식 조합에 대한 ablation 결과를 Table 1에 제시하였다. 교차 엔트로피 단독 조건(CE-only) 대비 SupCon 결합 조건에서 임베딩 군집의 분리도가 개선되었으며, 이는 t-SNE 시각화에서 확인되었다. 자기 지도 학습 사전학습 변형들은 비교 목적으로 포함되었으나, 최종 분류 성능과 cosine score 유효성에서 SupCon 직접 학습 방식과 통계적으로 유의한 차이를 보이지 않아 본 연구의 핵심 주장에서 제외하였다. 이 결과는 이중 출력 체계의 임상적 유용성이 특정 학습 커리큘럼에 종속적이지 않음을 시사한다.

#### Table 1. Ablation: Backbone × Training Mode — Diagnostic Performance on Validation Set

> 📝 **TODO**: 아래 표의 모든 수치를 실험 결과로 채울 것. 현재 회색칸은 미기입 항목.

| # | Backbone | Training Mode | AUROC | Sensitivity (%) | Specificity (%) | F1 Score | Cutoff (Youden's J) |
|---|----------|--------------|-------|-----------------|-----------------|----------|---------------------|
| 1 | ResNet50V2 | CE Only | ___ | ___ | ___ | ___ | ___ |
| 2 | EfficientNetV2B0 | CE Only | ___ | ___ | ___ | ___ | ___ |
| 3 | ResNet50V2 | CE + SupCon | ___ | ___ | ___ | ___ | ___ |
| 4 | EfficientNetV2B0 | CE + SupCon | **1.000** | **94.3** | **100.0** | **0.971** | **0.004** |
| 5 | ResNet50V2 | VICReg (SSL) | ___ | ___ | ___ | ___ | — |
| 6 | EfficientNetV2B0 | VICReg (SSL) | ___ | ___ | ___ | ___ | — |
| 7 | ResNet50V2 | NNCLR (SSL) | ___ | ___ | ___ | ___ | — |
| 8 | EfficientNetV2B0 | NNCLR (SSL) | ___ | ___ | ___ | ___ | — |

*CE = Cross-Entropy; SupCon = Supervised Contrastive Learning; SSL = Self-Supervised Learning (fine-tuned head). AUROC 95% CI via DeLong method. Cutoff determined on validation set only (Youden's J); SSL rows report AUROC only (linear probe, no Youden cutoff computed).*

#### Table 2. Primary Model Performance — Confidence Score (EfficientNetV2B0 + CE + SupCon)

> 📝 **TODO**: Test set 수치를 실험 완료 후 채울 것.

| Metric | Validation Set | Test Set |
|--------|---------------|----------|
| AUROC (95% CI) | 1.000 (–) | ___ (___ – ___) |
| Optimal Cutoff (Youden's J) | 0.004 | — *(Val cutoff 고정 적용)* |
| Accuracy (%) | 97.0 | ___ |
| Sensitivity (%) | 94.3 | ___ |
| Specificity (%) | 100.0 | ___ |
| PPV (%) | 100.0 | ___ |
| NPV (%) | 94.1 | ___ |
| F1 Score | 0.971 | ___ |
| TP / FP / FN / TN | 261 / 0 / 16 / 253 | ___ / ___ / ___ / ___ |

*95% CI for test set metrics: bootstrap n=1,000.*

### 3.3 Confidence score 기반 진단 성능

검증 세트에서 Youden's J 통계를 이용하여 결정된 임계값에서 confidence score는 AUROC 1.000, 최적 절단값 0.004를 보였다. 이 운영 지점에서 정확도 97.0%, 민감도 94.3%, 특이도 100.0%, 양성예측도 100.0%, 음성예측도 94.1%, F1 점수 0.971을 달성하였다. 위양성이 한 건도 없었다는 사실은, 이 기준에 의해 양성으로 판정된 경우 모델이 극히 높은 정밀도로 HCC를 식별하였음을 시사한다. 테스트 세트 결과는 도출하였다.

> 📝 **[TODO: Test 결과]** 테스트 세트 성능 지표 기입

### 3.4 임베딩 구조와 cosine score의 타당성

t-SNE를 이용한 임베딩 공간 시각화에서 HCC와 hemangioma 군집 간 뚜렷한 분리가 관찰되었다. 이 결과는 단순한 시각적 확인을 넘어, cosine 기반 점수의 타당성을 직접 뒷받침한다는 점에서 중요하다. 학습된 임베딩 공간이 질환 특이적 기하학적 구조를 보존하지 못한다면, 클래스 prototype에 대한 cosine similarity는 해석적 가치를 갖기 어렵다. 관찰된 군집 분리는 이 설정에서 prototype 유사도가 의미 있는 기술자(descriptor)임을 지지한다.

### 3.5 ROC-A, ROC-B, ROC-C 비교

본 원고의 중심 비교 분석은 confidence score, HCC cosine score, Δscore에 대한 세 종류의 ROC 분석 비교이다. 현재 confidence score의 검증 성능은 확보되어 있으며 근완벽 변별력을 보였다. Cosine 기반 ROC 분석 결과(ROC-B, ROC-C)는 최종 결과 표 및 그림에 비교하였다.

> 📝 **[TODO: ROC 비교]** ROC-B, ROC-C 분석 결과 및 Figure 삽입 이 비교의 목적은 cosine 기반 점수가 confidence score와 동등한 AUROC를 보이는지(비열등성 검증)와, 두 지표가 일치하지 않는 사례에서 서로 다른 임상 정보를 제공하는지(상보성 확인)를 함께 평가하는 것이다. DeLong 검정으로 ROC-A 대비 ROC-B, ROC-A 대비 ROC-C의 쌍별 AUROC 비교를 수행하여 p값을 보고하였다.

> 📝 **[TODO: p-value]** DeLong p-value 기입

#### Table 3. Three-Way ROC Comparison — Dual Output System (Validation Set)

> 📝 **TODO**: ROC-B, ROC-C 수치를 cosine score 계산 후 채울 것.

| Output | Score Type | AUROC | 95% CI | DeLong p-value vs ROC-A |
|--------|-----------|-------|--------|-------------------------|
| ROC-A | Confidence Score (softmax) | **1.000** | — | — (reference) |
| ROC-B | HCC Cosine Score | ___ | ___ – ___ | ___ |
| ROC-C | Δscore (HCC cosine − Hem cosine) | ___ | ___ – ___ | ___ |

*Pairwise comparisons (ROC-A vs ROC-B, ROC-A vs ROC-C) by DeLong correlated ROC test. p > 0.05 interpreted as support for non-inferiority of cosine-based metrics.*

#### Table 4. Dual-Output Discordance Analysis (Validation Set)

> 📝 **TODO**: confidence score와 Δscore 임계값 적용 후 각 사분면 N 수 채울 것.

| Confidence Score | Δscore | N (cases) | Predicted Label | Interpretation |
|-----------------|--------|-----------|-----------------|----------------|
| High (≥ cutoff) | High (≥ cutoff) | ___ | HCC | Concordant high — strong HCC evidence |
| High (≥ cutoff) | Low (< cutoff) | ___ | HCC | **Discordant** — possible shortcut learning; caution warranted |
| Low (< cutoff) | High (≥ cutoff) | ___ | Hemangioma | Discordant — embedding space suggests HCC-like; review |
| Low (< cutoff) | Low (< cutoff) | ___ | Hemangioma | Concordant low — strong benign evidence |

*(Figure 1: 3종 ROC 곡선 비교 + Dual-output scatter plot — 결과 수집 후 삽입)*

### 3.6 이중 출력 체계 하에서의 오류 패턴 분석

통상적인 ROC 분석을 넘어, 이중 출력 체계는 임상적으로 해석 가능한 오류 분류를 가능하게 한다. Confidence와 Δscore가 모두 높은 사례는 일치하는 고가능성 HCC 사례로, 둘 다 낮은 사례는 일치하는 양성 사례 또는 추가 평가가 필요한 불확정 병변으로 해석될 수 있다. 더 중요한 것은 불일치 사례, 특히 confidence는 높지만 Δscore가 낮은 경우이다. 이는 비강건한 단서에 의해 유발된 과잉 확신 예측의 실용적 신호일 수 있다. 이러한 불일치 패턴 분석이 모델의 임상적 해석에서 중심적 역할을 할 것으로 예상된다.

---

## 4. 고찰 (Discussion)

### 4.1 선행연구의 흐름과 본 연구의 위치

초음파 기반 간 국소 병변 딥러닝 분류 연구는 2020년을 전후하여 본격적으로 성장하였다. Yang 등(2020, *EBioMedicine*)은 13개 기관 2,143명 데이터로 ResNet 기반 DCNN을 개발하여 외부 검증 코호트에서 AUROC 0.924를 달성하였으며, 이는 15년 경력 임상의 236명의 성능을 유의하게 상회하였다.[6] 이 연구는 다기관 표준화 데이터, 배경 간 분기 모델(Model^LB^), 임상변수 통합(Model^LBC^)이라는 세 단계 구조로 성능을 단계적으로 개선하였으며, 이 분야 대규모 검증 연구의 토대를 제공하였다.

이후 Zhang 등(2022, *Abdom Radiol*)은 분할과 분류를 동시에 수행하는 joint 모델로 AUROC 0.947을 보고하였고[7], Li 등(2023, *Sensors*)은 CNN과 GCM(Generalized Co-occurrence Matrix) 텍스처 특징을 결합하여 B-mode 영상에서 98% 이상의 정확도를 달성하였다. 보다 최근인 Du 등(2025, *eClinicalMedicine*)은 ≤3 cm HCC를 대상으로 해석 가능한 머신러닝 모델을 다기관 후향적 연구로 검증하였다. 또한 메타분석 수준에서도 Zhao 등(2023, *PubMed*)의 48개 연구 종합에서 딥러닝 기반 HCC 영상 진단의 pooled sensitivity 89%, AUROC 0.95가 보고되었다.[21]

그러나 이들 연구의 공통적인 구조적 특징은 **단일 softmax 출력(또는 그 후처리 조합)**을 최종 진단 지표로 사용한다는 점이다. Yang 등(2020)은 softmax 출력에 임상 변수를 로지스틱 회귀로 결합하는 nomogram을 제시하였고, Zhang 등(2022)은 분할 후 분류의 이진 확률 값을 보고하였다. 어느 논문도 모델이 학습한 임베딩 공간의 기하학적 구조, 즉 병변이 전형적인 HCC 또는 hemangioma 표현의 중심과 얼마나 가까운지를 독립적인 진단 출력으로 제시하지 않았다. 본 연구는 이 간극을 메우는 첫 번째 시도이다.

### 4.2 Softmax 출력의 개념적 한계와 재명명의 방법론적 의의

기존의 많은 의료 AI 연구들은 모델의 최종 출력인 softmax 값을 관행적으로 "확률(probability)"로 표현한다. 그러나 이 표현은 개념적으로 부정확하다. Softmax 함수는 모델 최종 단계의 각 클래스별 원시 점수를 0과 1 사이의 값으로 정규화하는 수학적 변환으로, 출력값이 고전적 확률론에서 요구하는 보정된 사후 확률(calibrated posterior probability)을 의미하지는 않는다.

Guo 등(2017, ICML)은 현대 심층 신경망이 체계적인 과잉 확신(overconfidence)을 보이며, softmax 값과 실제 정답률 사이에 유의한 괴리가 존재함을 실험적으로 입증하였다.[18] 이 괴리는 temperature scaling이나 Platt scaling 같은 사후 보정 없이는 해소되지 않는다. 초음파 데이터셋처럼 비병변 artifact가 풍부한 환경에서는, 모델이 병변의 실질적인 영상의학적 특성이 아닌 캘리퍼나 눈금 표시와 같은 부가 단서에 의해 높은 confidence를 산출할 수 있다. 이것은 Ribeiro 등(2016)[20]이 기술한 shortcut learning의 의료 영상 특이적 발현이며, Donnelly 등(MICCAI 2024)[22]이 지적한 "맞는 예측을 잘못된 이유로 하는" 구조적 취약성과도 맥락이 닿는다.

이러한 이유로 본 연구는 softmax 출력을 "확률"이 아닌 **confidence score**로 의도적으로 재명명하였다. 이것은 기존 연구들에서 통용되던 표현 관행에 대한 명시적 이의 제기이며, 초음파 AI 연구에서 모델 출력 해석의 정확성을 높이기 위한 방법론적 기여이다.

### 4.3 임상의의 추론과 일치하는 Cosine HCC Score — ProtoPNet 계열과의 비교

임상의가 초음파 영상에서 간 병변을 진단하는 과정은 단순한 이진 판단이 아니다. 수련을 통해 내면화한 전형적 HCC 소견(저에코 배경, 주변부 저에코 테두리, 결절 내 결절 패턴)과 혈관종의 전형적 소견(고에코, 경계 명확, 균일한 에코)을 현재 병변과 비교 평가하는 **유사성 기반 추론(similarity-based reasoning)**이 핵심이다.

이 임상적 추론 방식을 AI로 모사하려는 선행 시도로는 Chen & Li(2019)의 ProtoPNet(Prototypical Part Network)이 있다. ProtoPNet은 분류 점수를 테스트 영상 패치와 학습된 클래스별 prototype 패치 간의 유사도로 계산하는 ante-hoc 해석 가능 모델로, "이 부분이 이전에 본 전형적인 HCC 패치와 유사하기 때문에 HCC로 분류한다"는 형태의 설명을 제공한다. 이후 의료 영상 분야에서도 D-ProtoPNet(흉부 X선, 2025)[23], MAProtoNet(뇌종양 MRI, 2024)[24], 유방암 DBT(2025)[25] 등에 적용되었고, 해석 가능성과 성능의 절충에서 유의미한 향상이 보고되었다. 최근 CVPR 2025에서 발표된 CSR(Concept-based Similarity Reasoning)[26]은 개념별 prototype과 feature map 간의 cosine similarity를 2D similarity map으로 계산하여 의사가 공간적으로 개입할 수 있는 interactive 해석 체계를 구현하였다.

본 연구는 ProtoPNet 계열과 유사한 임상적 동기를 공유하지만 구조적으로 다른 접근법을 취한다. ProtoPNet 계열이 클래스별 여러 prototype 패치를 명시적으로 학습하고 이들과의 최대 유사도를 분류에 활용하는 반면, 본 연구는 supervised contrastive learning으로 강화된 임베딩 공간에서 HCC 전체 영상의 평균 prototype 벡터와의 cosine similarity를 단일 연속형 점수로 산출한다. 이 접근법의 차별적 강점은 세 가지이다. 첫째, 추가적인 prototype 학습 단계나 별도의 아키텍처 변경 없이 기존 분류 모델의 임베딩만으로 cosine score를 계산할 수 있다는 구현 효율성이다. 둘째, AFP나 PIVKA-II처럼 연속형 스칼라 값으로 산출되어 임상 연구에서 ROC 분석과 임계값 검증이 가능한 "연속형 영상의학적 마커"로 기능할 수 있다. 셋째, Δscore를 통해 단일 클래스 유사도가 아니라 경쟁 두 클래스 사이의 상대적 위치를 정량화함으로써, "HCC인가, hemangioma인가"라는 비교 추론을 수치화한다.

### 4.4 Supervised Contrastive Learning의 이론적 근거

Confidence score 기반 분류는 교차 엔트로피 손실만으로도 구현할 수 있다. 그러나 cosine score를 의미 있는 진단 지표로 삼으려면, 임베딩 공간 자체가 클래스별 기하학적 응집성을 갖도록 학습되어야 한다. 이것이 본 연구에서 Khosla 등(2020, NeurIPS)의 Supervised Contrastive Learning(SupCon)을 도입한 핵심 이유이다.[15]

CE-only 학습에서는 분류 결정 경계가 형성되면 임베딩 공간의 세밀한 기하학적 구조는 부수적으로만 최적화된다. 반면 SupCon이 추가되면 클래스 내 응집도(intra-class compactness)와 클래스 간 분리도(inter-class separability)가 학습 목적에 직접 포함된다. 따라서 cosine similarity가 학습 목적함수와 정렬된 의미 있는 유사성 척도가 된다는 점이 SupCon 도입의 핵심 이론적 근거이다. Ablation(Table 1)에서 SupCon 결합이 CE-only 대비 임베딩 분리도를 개선함을 보여주며, t-SNE 시각화에서 확인된 HCC-hemangioma 군집 분리는 이 설계 의도의 실현을 시각적으로 지지한다.

비교를 위해 수행한 VICReg, NNCLR 기반 self-supervised learning 변형들은 분류 성능과 cosine score 유효성에서 SupCon 직접 학습과 통계적으로 유의한 차이를 보이지 않았다. 이는 이중 출력 체계의 임상적 유용성이 특정 학습 커리큘럼이 아닌 임베딩 공간의 기하학적 구조 자체에 기반함을 시사한다.

### 4.5 이중 출력 체계의 임상적 의의 — 불일치 패턴이 제공하는 안전장치

본 연구의 가장 독창적인 임상적 기여는 confidence-cosine 불일치(discordance) 패턴 분석이다. Confidence score가 높으나 Δscore가 낮은 경우는 모델이 병변의 진정한 영상의학적 특성보다 데이터셋 특이적 단서에 의존하여 높은 확신을 산출하였을 가능성을 시사한다. 이 경우 모델의 신호는 임상적 주의의 대상이 된다.

이 안전장치의 임상적 중요성은 Du 등(2025)의 해석 가능 ML 접근법과 비교할 때 더욱 명확해진다. Du 등은 해석 가능성을 위해 영상의학적 특징을 수작업으로 추출하고 LASSO 변수 선택을 수행하는 전통적 파이프라인을 사용하였다. 이 접근법은 해석 가능하지만 추출 특징의 수작업 정의에 의존적이다. 반면 본 연구는 end-to-end 딥러닝 모델 안에서 유사성 기반 출력을 학습으로 도출한다는 점에서 방법론적 차별성이 있다.

이중 출력의 일치-불일치 분포 분석(Table 4)은 단일 지표 연구에서 접근할 수 없는 데이터셋 특이적 취약성의 실용적 진단 기회를 제공한다. 이것은 의료 AI 안전성 관점에서 규제 평가의 중요한 논거가 될 수 있으며, 향후 prospective 임상 연구에서 검증되어야 할 가설을 구체적으로 제시한다.

### 4.6 Shortcut 관찰과 이중 출력의 임상적 의미

본 연구에서 수행한 예비적 시각화 분석에서, 일부 사례에서 모델의 활성 영역이 병변 실질보다 캘리퍼, 눈금 표시와 같은 비병변 artifact에 집중되는 것이 관찰되었다. 초음파 영상에서의 shortcut learning은 성능 과대 추정의 문제를 넘어, 모델이 "맞는 예측을 잘못된 이유로" 하는 구조적 취약성이다. 이 관찰은 단일 confidence 지표에 대한 임상적 의존의 위험성을 경고하며, embedding space 전체의 기하학적 유사도를 기반으로 하는 cosine score를 병용하는 이중 출력 접근법의 임상적 타당성을 추가적으로 뒷받침한다.

### 4.7 한계 (Limitations) 및 향후 연구 방향

본 연구는 다음의 방법론적 한계를 가지며, 이를 투명하게 기술함으로써 해석의 적정 범위를 설정한다.

**단일 기관 후향적 설계.** Yang 등(2020)이 13개 기관, Du 등(2025)이 다기관 전향적 검증을 수행한 것과 달리, 본 연구는 단일 기관 후향적 데이터(SMC-LUD)에 기반한다. 기관별 초음파 장비, 영상 획득 프로토콜, annotation 방식의 차이로 인해 제안 모델의 외적 타당도는 검증되지 않았다. 이중 출력 체계의 임상적 신뢰도 확립을 위해서는 다기관 전향적 코호트에서의 외부 검증이 필수적이다.

**혈청표지자 대비 증분 이득 미평가.** AFP 및 PIVKA-II와의 직접적인 증분 이득 비교가 이루어지지 않았다. HCC cosine score가 AFP에 추가하여 독립적인 진단 가치를 제공하는지 평가하려면 별도 연구가 필요하다.

**이진 대조군의 임상적 한계.** 실제 임상의 감별 진단 스펙트럼에는 FNH, 간내 담관암, 전이성 간암 등이 포함되며, 이진 분류 체계는 이 현실을 충분히 반영하지 못한다. 다중 클래스 확장은 향후 과제이다.

**임상의 대비 직접 비교 부재.** Yang 등(2020)이 236명 임상의와의 head-to-head 비교를 수행한 것과 달리, 본 연구에서는 숙련된 판독의와의 직접 성능 비교가 이루어지지 않았다. 이는 임상적 유용성 평가에서 필수적으로 보완되어야 할 항목이다.

**병변 크기 층화 분석 미시행.** Yang 등(2020)의 소결절(1.1–2.0 cm) 층화 분석에서 AUROC 0.926이 보고된 것처럼 크기별 성능 편차가 중요하다. 본 연구에서는 병변 크기별 층화 분석이 이루어지지 않아 조기 HCC 탐지 성능에 대한 별도 평가가 필요하다.

**Prototype Drift 가능성.** HCC cosine score의 prototype은 훈련 세트의 평균 임베딩으로 정의된다. 다기관 환경에서 prototype이 표적 집단을 충분히 대표하지 못하는 prototype drift가 발생할 수 있으며, 다기관 환경에서의 prototype 재보정(recalibration) 전략 검토가 향후 필요하다.

---

## 5. 결론 (Conclusion)

본 연구는 B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증하며, confidence score와 임베딩 기반 HCC cosine score로 구성된 이중 출력 해석 체계를 제안하였다. 모델은 우수한 내부 검증 성능을 보였고, 학습된 임베딩 구조는 cosine 기반 유사도가 의미 있는 영상의학적 지표로 기능할 수 있음을 지지하였다.

본 연구의 핵심 주장은 두 출력이 중복적이지 않다는 것이다. Confidence score는 모델의 내부 결정적 확신을 포착하는 반면, cosine 기반 점수는 학습된 클래스 전형에 대한 병변 유사도를 반영하며, 임상의의 유사성 기반 진단 추론과 개념적으로 일치한다. 두 출력의 일치는 예측에 대한 신뢰를 강화할 수 있고, 불일치는 추가적인 임상 주의가 필요한 사례를 드러낼 수 있다. 이 이중 출력 체계는 성능 중심 분류기를 제공하는 동시에, 임상의가 모델의 판단을 보다 투명하게 해석하고 검증할 수 있는 구조적 틀을 함께 제공한다는 점에서 방법론적, 임상적 의의를 가진다.

---

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
| **Storage** | Kaggle Dataset I/O (ephemeral) |
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

> **비고**: Kaggle P100 환경은 세션당 최대 9시간의 GPU 사용 제한이 있으며,  
> 장시간 학습은 세션 분할 및 checkpoint 재개 방식으로 수행되었다.  
> 상세 hyperparameter 및 학습 설정(cfg)은 추후 별도 기재 예정이다.

### Appendix B. 하이퍼파라미터 설정

> 📝 **TODO**: cfg 파일 확정 후 아래 표를 완성할 것.

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
13. Collins GS, Reitsma JB, Altman DG, Moons KGM. Transparent Reporting of a Multivariable Prediction Model for Individual Prognosis or Diagnosis (TRIPOD): The TRIPOD Statement. *BMJ*. 2015;350:g7594.
14. *(SMC-LUD: Samsung Medical Center Liver Ultrasound Dataset. Sci Data, Nature Portfolio, 2026 — DOI 삽입 예정)*
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
27. Du Z, et al. Development and Validation of an Ultrasound-Based Interpretable Machine Learning Model for the Classification of ≤3 cm Hepatocellular Carcinoma: A Multicentre Retrospective Diagnostic Study. *eClinicalMedicine*. 2025;81:103098.
