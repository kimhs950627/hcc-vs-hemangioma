# 논문 초고 개요 — 의학 저널 투고용

> **작성 상태**: 전체 한글 서술체 초고 (rev. 2026-06-29)
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

B-mode 초음파는 간세포암(hepatocellular carcinoma, HCC) 감시에서 가장 널리 사용되는 1차 영상 도구이지만, 조기 병변에 대한 민감도는 제한적이며 검사자 의존성이 크다. 최근 딥러닝 기반 초음파 분류 모델들이 보고되고 있으나, 대부분 softmax 출력값을 "확률"로 해석하는 단일 출력 체계를 유지하고 있으며, 임상가가 판독에 사용하는 유사성 기반 추론과의 연결은 충분히 제시되지 못하였다. 특히 초음파 영상에는 캘리퍼, 측정 눈금, 텍스트 annotation 등 비병변성 artifact가 존재하여 모델의 shortcut learning을 유발할 수 있으며, 이 경우 높은 confidence만으로는 임상적 신뢰를 제공하기 어렵다.

### 방법 (Methods)

본 연구는 단일 기관 후향적 연구로, 삼성서울병원에서 구축된 공개 B-mode 간 초음파 데이터셋(SMC-LUD)을 사용하였다. 환자 단위 분리를 유지한 상태에서 훈련, 검증, 테스트 세트로 영상을 구분하였다. 모델은 CNN backbone과 transformer encoder를 결합한 hybrid vision transformer로 설계하였고, 교차 엔트로피 단독 학습과 supervised contrastive learning(SupCon)을 결합한 학습을 비교하였다. 단일 모델에서 두 가지 출력을 정의하였다. 첫째, softmax 출력값은 고전적 의미의 확률이 아닌 모델의 상대적 확신 정도를 나타내는 **confidence score**로 정의하였다. 둘째, 임베딩 공간에서 HCC prototype과의 cosine similarity를 이용한 **HCC cosine score**를 산출하였으며, hemangioma prototype과의 차이를 반영한 **Δscore**도 추가 계산하였다. 최적 임계값은 Youden's J 통계를 이용하여 검증 세트에서만 결정하였고, 테스트 세트는 맹검 평가에만 사용하였다.

### 결과 (Results)

검증 세트에서 confidence score 기반 분류는 AUROC 1.000, 정확도 97.0%, 민감도 94.3%, 특이도 100.0%, 양성예측도 100.0%를 보였다. t-SNE 시각화에서 HCC와 hemangioma의 임베딩 군집이 뚜렷하게 분리되었으며, 이는 cosine similarity 기반 점수가 의미 있는 표현 공간 위에서 산출됨을 시사하였다. Confidence score, HCC cosine score, Δscore 각각에 대한 ROC 분석을 비교하여, softmax confidence와 embedding similarity가 서로 다른 진단 정보를 제공할 수 있음을 확인하고자 하였다.

### 결론 (Conclusions)

Hybrid vision transformer 기반 이중 출력 접근법은 B-mode 초음파만으로도 높은 변별력을 보였으며, softmax confidence에 더해 cosine similarity 기반 HCC score를 병용함으로써 기존 CAM 기반 설명의 한계를 보완할 수 있는 가능성을 제시하였다. 이 이중 출력 체계는 모델의 예측을 단일 스칼라 값으로 환원하지 않고, 임상가가 실제로 사용하는 "전형적 병변과의 유사성"이라는 해석 축을 함께 제공한다는 점에서 방법론적, 임상적 의의를 가진다.

**핵심어**: 간세포암; 초음파; 딥러닝; supervised contrastive learning; vision transformer; cosine similarity; 보정(calibration); 설명 가능 인공지능

---

## 1. 서론 (Introduction)

간세포암은 원발성 간암의 가장 흔한 조직학적 아형으로, 전 세계적으로 암 관련 사망의 주요 원인 중 하나이다. 임상적으로 예후는 진단 시 병기에 크게 의존하는데, 조기 병변은 수술적 절제, 고주파 열치료, 간이식 등 완치적 치료가 가능한 반면, 진행 병변은 생존율이 현저히 낮다.[1][2] 이러한 이유로 국내외 진료 지침은 고위험 환자군에서 복부 초음파를 이용한 정기적 감시 검사를 권고하고 있다.[2][3]

B-mode 초음파는 현재 이용 가능한 감시 도구 중 가장 접근성이 높고 임상 현장에서 가장 광범위하게 사용되는 영상 방법이다. 비침습적이고, 비용이 저렴하며, 반복 시행이 가능하고, 외래 환경에서 즉시 이용할 수 있다는 실용적 장점이 분명하다. 그러나 이러한 장점은 상당한 한계와 공존한다. 초음파만을 이용한 조기 HCC 감지 민감도는 여전히 만족스럽지 않으며, alpha-fetoprotein 등 혈청 바이오마커를 병용하더라도 성능 개선은 제한적이다.[4][5] 또한 초음파 검사의 진단 품질은 검사자의 경험, 스캔 기법, 판독 역량에 의해 크게 좌우된다. 이 검사자 의존성은 전문 영상의학과 의사가 상시 근무하지 않는 일차의료 및 지역사회 환경에서 특히 문제가 된다.

이러한 한계는 임상적으로 해결되지 않은 필요를 만들어낸다. 많은 일선 진료 현장에서 의사들은 간 병변이 조영증강 영상 검사 또는 전문의 평가로 이어질 만큼 충분히 의심스러운지를 판단해야 한다. B-mode 초음파에서 직접 진단 정보를 추출할 수 있는 딥러닝 모델은 이 맥락에서 유용한 보조 도구가 될 수 있다. 그러나 AI 시스템이 임상적으로 설득력을 가지려면 단순히 분류 성능이 높은 것으로는 충분하지 않다. 모델 출력이 임상가가 실제로 병변을 판단하는 방식과 일치하는 방향으로 해석될 수 있어야 한다.

최근 초음파 간 국소 병변 분류에서 딥러닝 기반 모델이 유망한 성능을 보인 연구들이 보고되었다.[6][7] 그러나 이 분야의 기존 문헌은 몇 가지 측면에서 한계를 지닌다. 첫째, 대부분의 모델은 단일 softmax 출력을 제공하며, 이를 "확률"로 기술하는 경향이 있으나, softmax 출력이 보정된(calibrated) 확률을 의미하지는 않는다.[18] 둘째, 임계값 설정이 최종 평가로부터 명확히 분리되지 않는 경우가 있어, 지나치게 낙관적인 성능 추정이 발생할 수 있다. 셋째, 많은 연구에서 CAM 기반 시각적 설명 도구를 활용하지만, 강조된 영역이 실제 병변 특성을 반영하는지 아니면 데이터셋 고유의 artifact를 반영하는지에 대한 논의가 충분하지 않다. 넷째, 비교적 소규모인 의료 영상 데이터셋에서 각각 고유한 단점을 가지는 기존 CNN 또는 순수 vision transformer 구조가 주로 사용되었다.[9][10][11]

초음파 영상에서 해석 가능성 문제는 특별한 강조가 필요하다. 전형적인 간 초음파 프레임에는 병변 자체와 무관한 정보, 즉 캘리퍼, 측정 눈금, 텍스트 오버레이, 기기 고유의 acquisition artifact 등이 포함될 수 있다. 모델이 이러한 불필요한 단서를 클래스 레이블과 연결하는 방식으로 학습하면, 겉으로는 우수한 분류 성능이 관찰되더라도 그것은 병변에 대한 실질적 이해가 아닌 shortcut learning을 반영하는 것일 수 있다. 이 경우 높은 softmax 출력은 임상적으로 유효한 영상 소견을 포착하였다는 증거가 아니라, 방사선학적으로 유효하지 않은 내부 패턴에 대한 확신을 반영할 뿐이다.[12][19][20]

이 구분이 중요한 이유는 방사선과 의사가 간 병변을 진단할 때 단순히 단일 스칼라 확신 값에 의존하지 않기 때문이다. 방사선과 의사는 목표 병변의 시각적 특성을 이전에 학습한 원형(prototype) 또는 패턴과 비교한다. 즉, 이 병변이 전형적인 HCC를 얼마나 닮았고, 양성 혈관종을 얼마나 닮았는가를 평가한다. 이러한 관점에서 유사성 기반 추론은 방사선 진단의 부수적 요소가 아니라 핵심 요소이다. Supervised contrastive learning은 같은 클래스의 병변을 임베딩 공간에서 군집시키고 다른 클래스를 분산시킴으로써 이러한 기하학적 구조를 학습하는 방법이다.[15] 이러한 표현 공간이 구성된다면, 클래스 prototype에 대한 cosine similarity는 단순한 후처리 수치 편의가 아니라 의미 있는 진단 지표로 기능할 수 있다.

이에 따라 본 연구는 두 가지 목적으로 설계되었다. 첫째, B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증한다. 둘째, 단일 출력 분류 체계를 넘어, confidence score와 임베딩 기반 HCC cosine score 및 Δscore로 구성된 이중 출력 체계를 제안한다. 이를 통해 유사성 기반 출력이 softmax confidence에 상호보완적인 임상 해석 정보를 제공할 수 있는지, 그리고 근완벽 분류 상황에서 shortcut 의존 confidence 신호에 대한 과잉 의존을 완화할 수 있는지를 평가하고자 하였다.

---

## 2. 대상 및 방법 (Materials and Methods)

### 2.1 연구 설계

본 연구는 대한민국 서울 삼성서울병원에서 수집된 B-mode 간 초음파 영상 아카이브를 이용한 단일 기관 후향적 관찰 연구이다. 본 논문은 예측 모델 연구에 대한 TRIPOD 보고 기준에 따라 작성되었다.[13] 기관생명윤리위원회(IRB) 승인번호 및 동의 면제 세부 내용은 최종 행정 확인 후 삽입될 예정이다.

### 2.2 데이터셋 및 연구 대상

본 연구에서는 삼성서울병원 간 초음파 데이터셋(Samsung Medical Center–Liver Ultrasound Dataset, SMC-LUD)을 사용하였다. SMC-LUD는 2015년부터 2024년까지 수집된 간 국소 병변의 B-mode 초음파 영상으로 구성된 공개 데이터셋으로, 병리학적으로 확인된 HCC 영상 2,716장과 영상 기준으로 진단된 혈관종(hemangioma) 영상 2,669장을 포함하여 총 1,021명 환자의 5,385장 흑백 영상으로 이루어져 있다.[14] 모든 영상은 384 × 384 픽셀의 흑백 부동소수점 배열로 표준화되었다.

본 분석에서는 동일 환자의 영상이 개발 단계와 평가 단계 사이에 교차 오염되지 않도록 환자 단위 분리를 적용하였다. 최종 데이터는 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구분되었다. HCC와 hemangioma의 클래스 비율은 세트 간에 균형 있게 유지되었으며, HCC가 전체의 약 52%, hemangioma가 약 48%를 차지하였다. 연령, 성별, 간경변 유무, 병변 크기 등의 임상 메타데이터는 완전한 데이터 연결이 확인되면 최종 원고에 통합될 예정이다.

### 2.3 모델 구조

제안 모델은 CNN backbone과 transformer encoder를 결합한 hybrid vision transformer이다. 이 설계는 두 패러다임의 상호 보완적 강점을 활용하기 위한 것이다. 합성곱 레이어는 초음파 영상에서 질감 및 경계 정보를 추출하는 데 적합한 국소 귀납 편향(local inductive bias)을 유지하는 반면, transformer 레이어는 특징 맵 전반에 걸친 전역 문맥 관계를 모델링할 수 있다.[9][10][11]

CNN backbone으로는 ImageNet 사전학습 가중치로 초기화된 ResNet50V2 또는 EfficientNetV2B0가 사용되었다. 이로부터 추출된 특징 맵은 패치 형태의 토큰으로 재형성되어 다양한 깊이의 transformer encoder에 입력되었다. 분류 토큰(classification token)이 부가되고, 그 최종 표현이 하위 예측에 사용되었다. 이 임베딩은 유사성 기반 후처리 점수 계산에도 동일하게 활용되어, 모델이 단일 통합 체계 내에서 기존 분류 출력과 임베딩 기반 방사선학적 유사도 출력을 동시에 생성할 수 있도록 하였다.

### 2.4 학습 전략

두 가지 학습 조건을 비교하였다. 첫 번째 조건에서는 교차 엔트로피 손실(cross-entropy loss)만을 이용하여 모델을 최적화하였다. 두 번째 조건에서는 교차 엔트로피 손실에 supervised contrastive loss를 결합하였다. 교차 엔트로피는 이진 분류를 위한 표준적인 판별 학습 신호를 제공하는 반면, supervised contrastive learning은 같은 클래스의 표본을 임베딩 공간에서 가깝게, 다른 클래스의 표본을 멀게 배치하도록 표현 공간을 명시적으로 구조화한다.[15]

교차 엔트로피 손실은 예측 클래스 분포와 원-핫 인코딩된 목표 레이블을 이용하여 통상적인 방식으로 정의하였다. Supervised contrastive loss는 Khosla 등의 정식화를 따랐으며, 정규화된 임베딩에서 같은 클래스에 속하는 표본은 양성 쌍으로, 배치 내 나머지 표본은 음성 쌍으로 처리하였다.[15] 대조 학습 조건에서 전체 손실은 교차 엔트로피와 supervised contrastive loss의 가중 합으로 구성되었다.

이 학습 방식의 선택이 갖는 개념적 중요성은 학습된 표현의 기하학적 구조에 있다. 교차 엔트로피만으로 학습할 때 임베딩 공간은 분류 목적을 통해 간접적으로 최적화되며, 표본 간 기하학적 거리가 의미를 갖도록 보장되지 않는다. 반면 supervised contrastive learning을 적용하면, 동일 클래스 병변이 임베딩 공간에서 인접한 영역을 차지하고 다른 클래스 병변은 분산되도록 학습 목적 자체가 설계된다. 이로 인해 cosine similarity는 선택적인 후처리 분석 도구가 아니라, 학습 목적이 명시적으로 의미 있도록 독려한 유사성의 척도가 된다. 본 연구에서 이것이 HCC cosine score와 Δscore를 구조화된 출력으로 정의하는 근거가 된다.

모든 모델은 384 × 384 흑백 입력 영상을 사용하여 학습되었다. 옵티마이저는 Adam을 사용하였고, 초기 학습률은 1 × 10⁻⁴였으며, 학습률 스케줄링은 warmup이 적용된 cosine annealing 방식을 따랐다. 데이터 증강에는 수평 및 수직 반전, 무작위 자르기 및 크기 조정, 밝기 및 대비 변환, Gaussian blur가 포함되었다. 학습은 클라우드 GPU 환경에서 실용적 계산 제약 조건 내에서 수행되었다.

### 2.5 이중 출력 지표 정의

본 연구의 핵심은 두 가지 상이한 유형의 모델 출력을 명시적으로 구분하는 것이다. 첫 번째 출력인 **confidence score**는 HCC 클래스에 해당하는 softmax 값으로 정의하였다. 이 값은 0과 1 사이에 위치하지만, 자동적으로 진정한 확률로 해석되어서는 안 된다. 현대의 심층 신경망은 체계적으로 과잉 확신(overconfident)된 softmax 출력을 산출하는 것으로 알려져 있으며, 확률론적 해석이 정당화되기 위해서는 temperature scaling과 같은 사후 보정이 흔히 필요하다.[18] 이러한 이유로 본 연구에서는 softmax 출력을 "확률"이 아닌 **confidence score**로 의도적으로 명명하였다.

두 번째 출력인 **HCC cosine score**는 임베딩 공간에서 정의하였다. 모델 학습 완료 후, 훈련 세트 내 HCC 표본들의 평균 임베딩 벡터를 계산하여 HCC prototype을 구성하였고, hemangioma 표본들의 평균 임베딩 벡터로 hemangioma prototype을 구성하였다. 임의의 병변 영상에 대해 해당 임베딩 벡터와 HCC prototype 간의 cosine similarity를 계산함으로써 HCC cosine score를 산출하였다. 이 점수는 임베딩 공간에서 학습된 HCC 표현의 기하학적 중심과 해당 병변이 얼마나 가까운지를 반영하도록 설계되었다.

변별력을 더욱 정교화하기 위해 **Δscore**를 HCC cosine score와 hemangioma cosine score의 차이로 정의하였다. 이 차이는 HCC와의 유사성뿐 아니라 hemangioma로부터의 비유사성도 포착하도록 설계되었다. 실용적 관점에서 Δscore는 두 경쟁 질환 prototype 사이의 상대적 유사성 마진으로 이해할 수 있다.

이러한 이중 출력 구조는 임상적 추론에서 동기를 얻은 것이다. 방사선과 의사는 단순히 병변이 어떤 내부 확신 수준으로 악성인지를 묻는 것이 아니라, 해당 병변이 전형적인 HCC의 시각적 특성과 hemangioma와 같은 양성 유사 병변의 특성 중 어느 쪽을 더 닮았는지를 고려한다. 따라서 cosine 기반 출력은 인간의 패턴 비교에 개념적으로 더 가까운 표현을 제공하는 반면, confidence score는 모델 내부의 결정적 확신을 반영한다. 이 두 출력 간의 일치와 불일치를 연구하는 것이 본 연구의 주요 목적 중 하나이다.

### 2.6 임계값 결정 및 데이터 유출 방지

낙관적 편향을 최소화하기 위해 모든 운영 임계값은 검증 세트에서만 결정하였다. 주된 임계값 선택 방법은 Youden's J 통계를 이용하였으며, 이는 검증 세트 ROC 곡선에서 민감도와 특이도의 합에서 1을 뺀 값을 최대화하는 지점으로 정의하였다. 민감도 우선 전략을 부차적 분석으로 추가 고려하였다. 검증 세트에서 결정된 임계값은 테스트 세트에 변경 없이 고정 적용하였다. 이 절차는 confidence score, HCC cosine score, Δscore 각각에 대해 별도로 수행하였다.

### 2.7 시각화 및 설명 가능성 분석

모델의 의사결정 패턴을 조사하기 위해 Grad-CAM과 transformer attention 시각화를 보조적 설명 도구로 사용하였다.[12] Grad-CAM은 CNN backbone의 최종 합성곱 단계에 적용하여 예측 클래스에 강하게 기여한 영역을 나타내는 열지도를 생성하였다. Attention 시각화는 transformer encoder에서 생성하여 분류 토큰과 영상 토큰 사이의 상호작용을 검사하였다.

이러한 시각적 도구들은 의료 영상 독자들에게 친숙하다는 이유로 포함되었으나, 인과적 추론의 결정적 증거로 취급하지는 않았다. 강조 영역이 캘리퍼, 눈금 표시, 텍스트 annotation 등 비병변 구조물과 겹치는 사례에 특별한 주의를 기울였으며, 이러한 경우는 잠재적인 shortcut learning의 증거로 해석하였다. 이중 출력 체계는 시각적 설명의 대체물이 아니라, 높은 confidence 출력을 반드시 신뢰할 수 있다는 잘못된 해석에 대한 추가적 안전장치로 위치하였다.

### 2.8 통계 분석

모델 변별력은 주로 수신자 조작 특성 곡선(ROC curve) 아래 면적(AUROC)을 이용하여 평가하였다. AUROC의 신뢰구간은 DeLong 방법으로 추정할 예정이다.[16] 세 가지 유형의 ROC 분석을 계획하였다. ROC-A는 confidence score를, ROC-B는 HCC cosine score를, ROC-C는 Δscore를 기준으로 한다. 이 세 ROC 곡선 간의 쌍별 비교는 DeLong 검정을 이용한 상관 ROC 비교로 수행할 예정이다.

임계값 의존적 성능 지표로는 민감도, 특이도, 양성예측도, 음성예측도, F1 점수 및 혼동행렬이 포함되었다. 또한 이중 출력 산점도를 통해 confidence score와 Δscore의 결합 분포를 시각화하고, 불일치 패턴에 특별한 주의를 기울일 예정이다. 잠재적 임상 순편익을 평가하기 위한 의사결정 곡선 분석도 계획하였다.[17] 테스트 세트 지표의 신뢰구간 추정을 위해 1,000회 반복 부트스트랩 재표본 추출을 사용할 예정이다.

---

## 3. 결과 (Results)

### 3.1 데이터셋 구성

분석 데이터셋은 훈련 세트 1,858장, 검증 세트 530장, 테스트 세트 268장으로 구성되었다. 전체 세트에 걸쳐 HCC와 hemangioma의 비율은 안정적으로 유지되었으며, HCC가 근소하게 많았다. 이 클래스 균형은 극단적인 클래스 불균형으로 인해 성능 지표가 왜곡될 위험을 줄였다. 메타데이터 가용성 확인 후 환자 인구통계학적 정보 및 병변 수준 요약 표를 추가할 예정이다.

### 3.2 Confidence score 기반 진단 성능

검증 세트에서 Youden's J 통계를 이용하여 결정된 임계값에서 confidence score는 AUROC 1.000, 최적 절단값 0.004를 보였다. 이 운영 지점에서 정확도 97.0%, 민감도 94.3%, 특이도 100.0%, 양성예측도 100.0%, 음성예측도 94.1%, F1 점수 0.971을 달성하였다. 이 절단값에서 위양성이 한 건도 없었다는 사실은, 내부 검증 세트에서 이 기준에 의해 양성으로 판정된 경우 모델이 극히 높은 정밀도로 HCC를 식별하였음을 시사한다.

### 3.3 임베딩 구조와 cosine 점수의 타당성

t-SNE를 이용한 임베딩 공간 시각화에서 HCC와 hemangioma 군집 간 뚜렷한 분리가 관찰되었다. 이 결과는 클래스 분리의 심미적 확인이라는 의미를 넘어서, cosine 기반 점수의 타당성을 직접 뒷받침한다는 점에서 중요하다. 학습된 임베딩 공간이 질환 특이적 기하학적 구조를 보존하지 못한다면, 클래스 prototype에 대한 cosine similarity는 해석적 가치를 갖기 어렵다. 관찰된 군집 분리는 이 설정에서 prototype 유사도가 의미 있는 기술자(descriptor)임을 지지한다.

### 3.4 ROC-A, ROC-B, ROC-C 비교 (계획)

본 원고의 중심 비교 분석은 세 가지 ROC 패러다임인 confidence score, HCC cosine score, Δscore 간의 대조이다. 현재 confidence score의 검증 성능은 확보되어 있으며 근완벽 변별력을 보였다. Cosine 기반 ROC 분석은 최종 결과 표 및 그림에 반영될 예정이다. 이 비교의 목적은 단순히 cosine 기반 점수가 AUROC에서 confidence score를 일치하거나 초과하는지를 판단하는 것에 그치지 않는다. 모델의 내부 확신이 불안정할 수 있는 사례에서 cosine 기반 점수가 상호보완적 정보를 제공하는지를 확인하는 것이 더 중요한 목적이다.

### 3.5 이중 출력 체계 하에서의 오류 패턴 분석

통상적인 ROC 분석을 넘어, 이중 출력 체계는 임상적으로 해석 가능한 오류 분류를 가능하게 한다. Confidence와 Δscore가 모두 높은 사례는 일치하는 고가능성 HCC 사례로 해석될 수 있다. 둘 다 낮은 사례는 일치하는 양성 사례 또는 추가 평가가 필요한 불확정 병변을 나타낼 수 있다. 더 중요한 것은 불일치 사례, 특히 confidence는 높지만 Δscore는 낮은 경우이다. 이는 비강건(non-robust) 단서에 의해 유발된 과잉 확신 예측을 나타낼 수 있다. 반대로 confidence는 낮지만 Δscore가 높은 사례는 분류기 출력이 더 신중함에도 불구하고 알려진 HCC prototype을 닮은 병변을 나타낼 수 있다. 이러한 불일치 패턴이 모델의 임상적 해석에서 중심이 될 것으로 예상된다.

### 3.6 설명 가능성 소견 및 Shortcut learning

예비적 Grad-CAM 검토에서 모델이 병변 실질 자체가 아닌 캘리퍼, 눈금 표시 등 비병변 artifact에 주의를 기울이는 것으로 보이는 사례가 확인되었다. 이는 모델의 외견상 확신 중 일부가 shortcut 특성과 연결될 수 있음을 시사하는 우려를 제기한다. 이 문제는 cosine 기반 출력을 보조적 해석 축으로 사용하는 동기가 되었다. 대표적 사례와 이에 해당하는 이중 출력 패턴은 최종 그림 세트에 포함될 예정이다.

---

## 4. 고찰 (Discussion)

본 연구는 AI 기반 간 초음파 분류 출력이 해석되어야 하는 방식에 대한 개념적 전환을 제안한다. Softmax 값을 자급자족적인 확률로 취급하는 대신, 본 연구는 모델의 결정적 확신을 반영하는 **confidence score**와, 훈련 데이터에서 학습된 클래스 특이적 기하학적 구조와 현재 병변이 얼마나 가까운지를 반영하는 **cosine 기반 질환 유사도 점수**를 명시적으로 구분한다. 이 구분은 단순히 명명의 문제가 아니라 방법론적 문제이다. 이는 의료 AI 문헌의 중심적 약점, 즉 경계화된 softmax 값이 임상적으로 해석 가능한 확률과 동등하다는 묵시적 가정을 다룬다.

첫 번째 주요 함의는 보정(calibration)에 관한 것이다. 현대의 심층 신경망은 과잉 확신된 출력을 산출하는 것으로 잘 알려져 있으며, 높은 softmax 값이 실제 정답률에 상응하지 않을 수 있다.[18] 의료 맥락에서 이것은 임상가들이 0.90 또는 0.95와 같은 출력을 신뢰할 수 있는 질환 확률로 자연스럽게 해석할 수 있기 때문에 중요하다. 이 값을 confidence score로 재명명함으로써, 본 연구는 모델이 실제로 학습한 것을 과장하는 것을 피한다. 이 개념적 명확화는 성능이 근완벽으로 보이는 시나리오에서 특히 중요한데, 점수를 과잉 해석하려는 유혹이 모델이 가장 인상적으로 보일 때 가장 강하게 나타나기 때문이다.

두 번째 함의는 해석 가능성에 관한 것이다. Grad-CAM과 같은 시각적 설명 방법은 여전히 유용하지만, 특히 shortcut learning이 비병변 artifact에서 발생할 수 있는 초음파 영상에서는 그 자체만으로는 충분하지 않다. 모델이 캘리퍼나 텍스트 오버레이를 강조하는 경우, 임상가는 예측이 진정한 병변 형태학에 의해 구동되는지 아니면 불필요한 acquisition 관련 단서에 의한 것인지라는 어려운 질문을 직면하게 된다. Cosine 기반 prototype 유사도는 이 문제를 완전히 해결하지는 못하지만, 다르고 임상적으로 직관적인 관점을 제공한다. 즉, 병변이 기하학적으로 알려진 HCC 사례들의 군집 가까이에 위치하는지를 보여준다. 그런 의미에서 이는 방사선과 의사의 추론과 일치하는 해석 지원의 한 형태를 제공한다.

세 번째 함의는 supervised contrastive learning의 역할에 관한 것이다. 통상적인 교차 엔트로피 학습에서 임베딩 공간은 분류 목적을 통해 간접적으로 최적화되며, 표본 간 기하학적 거리가 의미를 갖도록 보장되지 않는다. 이에 반해 supervised contrastive learning은 같은 클래스의 병변이 임베딩 공간에서 인접한 영역을 차지하고 다른 클래스 병변이 분산되도록 명시적으로 독려한다. 이로 인해 cosine similarity는 이론적으로 일관성 있는 유사도 척도가 되는데, 손실 함수 자체가 같은 클래스 내에서 높은 cosine similarity와 클래스 간에 낮은 cosine similarity를 선호하기 때문이다. 본 연구에서 이 속성이 HCC cosine score와 Δscore를 선택적 사후 분석이 아닌 구조화된 출력으로 정의하는 근거를 제공한다.

임상적 관점에서 제안된 이중 출력 체계는 confidence와 cosine similarity가 불일치하는 경우에 특히 유용할 수 있다. 이러한 불일치는 모델의 내부 확신이 질환 prototype 기하학에 의해 충분히 뒷받침되지 않는 사례를 표시할 수 있다. 실용적 관점에서 이는 의사들이 겉으로는 강한 AI 예측에 대해 과신을 삼가도록 도울 수 있으며, 필요한 경우 더 면밀한 검토나 추가 영상 검사를 독려할 수 있다. 따라서 이 체계는 블랙박스 출력에 대한 부적절한 의존에 대한 소박하지만 의미 있는 안전장치를 제공할 수 있다.

몇 가지 한계점을 언급해야 한다. 첫째, 본 연구는 후향적이며 단일 기관 데이터셋에 기반하여 외부 타당도가 제한된다. 둘째, 현재의 HCC 대 hemangioma 이진 비교는 실제 임상에서 접하는 간 국소 병변 감별의 전체 스펙트럼을 반영하지 못한다. 셋째, shortcut activation이 정성적으로 관찰되었으나, 완전한 정량적 artifact 위치 분석은 아직 완성되지 않았다. 넷째, 임상 메타데이터와 방사선과 의사 대비 비교가 현재 버전에 아직 통합되지 않았다. 마지막으로, prototype 기반 유사도 자체가 훈련 분포가 외부 집단을 충분히 대표하지 못하는 경우 기관 편향에 민감할 수 있다.

이러한 한계에도 불구하고 본 연구는 의료 AI에 잠재적으로 유용한 방법론적 기여를 제공한다. 초음파 분류기의 출력을 단일 확률 수치로 표현하는 방식을 넘어, 심층 네트워크의 수학적 속성과 방사선과 의사의 임상적 추론 과정 모두에 더 잘 부합하는 이중 축 의사결정 보조 체계로 재구성한다. 외부 검증이 이루어진다면, 이 체계는 초음파 기반 간 병변 평가에서 AI를 보다 해석 가능하고 신중하게 활용하는 것을 지원할 수 있을 것이다.

---

## 5. 결론 (Conclusion)

본 연구는 B-mode 초음파에서 HCC와 hemangioma를 감별하기 위한 hybrid vision transformer를 개발하고 검증하며, confidence score와 임베딩 기반 HCC cosine score로 구성된 이중 출력 해석 체계를 제안하였다. 모델은 우수한 내부 검증 성능을 보였고, 학습된 임베딩 구조는 cosine 기반 유사도가 의미 있는 방사선학적 지표로 기능할 수 있음을 지지하였다.

더 중요한 것은, 본 연구가 이 두 출력이 중복적이지 않음을 주장한다는 점이다. Confidence score는 모델의 내부 결정적 확신을 포착하는 반면, cosine 기반 점수는 학습된 클래스 prototype에 대한 병변 유사도를 반영한다. 두 출력의 일치는 예측에 대한 신뢰를 강화할 수 있고, 불일치는 더 큰 주의가 필요한 사례를 드러낼 수 있다. 이 이중 출력 체계는 따라서 성능 중심 분류기를 제공할 뿐 아니라, 미래의 초음파 AI 시스템을 위해 보다 임상적으로 일치된 해석 구조를 제공한다.

---

## 참고문헌

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
18. Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. *ICML*. 2017;70:1321–1330.
19. Nguyen A, Yosinski J, Clune J. Deep Neural Networks are Easily Fooled: High Confidence Predictions for Unrecognizable Images. *CVPR*. 2015:427–436.
20. Ribeiro MT, Singh S, Guestrin C. "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*. 2016:1135–1144.
