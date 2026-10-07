# 전체 원고 개편 및 리비전 이력 종합 보고서 (Comprehensive Revision History)

> **문서 목적**: 최초 원고(`manu_before_revision.docx`) 및 1차 수정본(`manuscript_revised.docx`) 대비 현재 **Revision 2** 최종본(`manuscript_revised_marked.docx`, `manuscript_revised_final.txt`)까지 이루어진 모든 학술적·통계적·문장 구조적 변경 사항을 종합 정리하고, 변경 사유와 논리적 근거를 기록함.

---

## 1. 개편 핵심 요약 (Executive Summary)

| 영역 | Revision 이전 (Original / Rev 1) | Revision 2 최종본 (Current) | 핵심 개편 사유 |
| :--- | :--- | :--- | :--- |
| **논문 제목** | "B-mode 복부 초음파에서 HCC 유사도를 정량화하는 이중 출력 영상표지자의 개발..." | "...딥러닝 모델의 Confidence Score와 임베딩 기반 유사도 점수: **단일기관 공개 데이터셋을 이용한 탐색적 연구**" | '개발' 및 '임상적 의의' 단정 등 과장된 표현을 지양하고, 탐색적 연구로서 연구 한계를 명확히 수용함 |
| **과제 정의** | 조기 간암 감시(Surveillance / Screening)와 2진 감별 진단을 혼용 서술 | 초음파상 우연히 발견된 결절에 대한 **"간세포암 vs 간혈관종 2진 감별 진단(Differential Diagnosis)"**으로 엄격히 한정 | 임상 과제 혼동으로 인한 심사위원 비판 사전 차단 |
| **데이터 규격** | "확진 시점에 근접한 대표 B-mode 영상 1장 선별"로 잘못 기술 | **문구 완전 삭제**. 744명 환자 대비 2,656장 영상(**환자당 평균 3.57장, 1:1 매칭 아님**) 실사 반영 및 한계점 명시 | 원저(SMC-LUD) 논문 실사 결과와 100% 정합성 유지 |
| **환자 분할** | 환자 간 분리 여부 모호 기술 | 원저의 **환자 단위 분할(70:20:10)** 준용으로 학습-검증 간 환자 누출(leakage) 배제 확증, 식별자 미제공 한계 서술 | 데이터 누출(Data leakage) 의혹 차단 및 엄밀성 확보 |
| **분류 성능 통계** | 점 추정치 위주, DeLong $p=1.000$으로 동등성 주장 | **Clopper-Pearson Exact Binomial 95% CI** 전면 병기, AUROC 1.000 천장 효과에 따른 DeLong 검정력 상실 인정 | 통계학적 엄밀성 확립 및 천장 효과 왜곡 방어 |
| **SupCon 효과 증명** | 단순 박스플롯(Fig 4, Fig 5) 및 중앙값 단순 비교 | **미학습 영상($n=798$) 대상 4패널 기하학적 표상 프로빙(Representation Probing)**: Wilcoxon 및 Pitman-Morgan 분산 축소 검정 도입 | ViT의 단순 암기(Shortcut)를 배제하고 SupCon의 필수 정규화 우위성을 수학적으로 입증 |
| **Softmax 해석** | "확률" 표현 혼용 | **별도 Calibration 부재 명시, 결정 강도(Confidence Score)로 재정의**, 흉부 X선/안저 영상 calibration 문헌([11, 16, 17]) 인용 | 머신러닝/통계 심사위원의 Calibration 개념 지적 원천 차단 |
| **연구 한계점** | 1~2줄의 형식적 기술 | **스펙트럼 편향, 배경 간 질환 미제공, 외부 검증 결여, 환자 내 상관성, 천장 효과 등 5대 한계점 심층 보강** | 비판적 피어 리뷰에 대비한 방어적 투명성 확보 |

---

## 2. 섹션별 상세 대조 및 변경 내역 (Section-by-Section Audit)

### 2.1 제목 및 저자 정보 (Title)
- **이전**: `B-mode 복부 초음파에서 HCC 유사도를 정량화하는 이중 출력 영상표지자의 개발: HCC Cosine Score와 Confidence Score의 상호보완적 임상적 의의`
- **Revision 2**: `B-mode 복부 초음파에서 간세포암과 간혈관종 감별을 위한 딥러닝 모델의 Confidence Score와 임베딩 기반 유사도 점수: 단일기관 공개 데이터셋을 이용한 탐색적 연구`
- **개편 근거**: 단일기관 공개 데이터셋 기반 후향적 연구에서 '영상표지자의 개발'이나 '상호보완적 임상 의의 확립'과 같은 단정적 어조는 심사위원의 즉각적 반발을 초래함. '탐색적 연구' 부제를 추가하여 톤을 학술적으로 낮춤.

---

### 2.2 초록 (Abstract)

#### 연구 배경 (Background)
- **이전**: 복부 초음파의 조기 HCC 감시 민감도(47%) 등을 언급하며 간암 감시(surveillance) 도구로 설명함.
- **Revision 2**: 우연히 발견된 간 국소 병변(focal liver lesion)에서 악성인 HCC와 양성인 간혈관종을 B-mode 단독으로 감별하는 임상적 어려움에 초점을 맞추도록 전면 수정.

#### 방법 (Methods)
- **이전**: Clean subset(744명, 2,656장)만 단순 나열.
- **Revision 2**: 전체 코호트(1,021명, 5,385장) 중 Clean subset(744명, 2,656장)의 원저 환자 단위 분할(훈련 1,858장, 검증 530장, 테스트 268장) 세부 수치를 정확히 명시. Youden's J 운영 임계값의 검증셋 독립 결정 및 테스트셋 고정 적용 명시.

#### 결과 (Results)
- **이전**: 검증셋 민감도 99.6%(실제 100.0% 오기재), 테스트셋 민감도 100.0%, 특이도 99.2%만 단순 보고. DeLong $p=1.000$ 주장.
- **Revision 2**:
  - 검증셋 민감도 100.0% (95% CI 98.7–100.0), 특이도 100.0% (95% CI 98.6–100.0), 정확도 100.0% (95% CI 99.3–100.0) 보고.
  - 테스트셋 민감도 100.0% (95% CI 97.4–100.0), 특이도 99.2% (95% CI 95.7–100.0, FP 1건, FN 0건), 정확도 99.6% (95% CI 97.9–100.0) 보고.
  - 테스트셋 오분류(FP=1)가 5가지 출력 모두에서 동일하여 단독 증분 이득(incremental benefit)을 입증하지 못했음을 정직하게 명시.
  - 미학습 실제 영상 대상 대응표본 통계 반영:
    - **True HCC ($n=417$)**: 중앙값 $0.882 \rightarrow 0.970$ 상승, 분산 $-60.7\%$ 축소 ($p < 0.001$).
    - **True Hemangioma ($n=381$)**: 중앙값 $0.955 \rightarrow 0.985$ 상승, 분산 $-98.7\%$ 축소 ($p < 0.001$).

#### 결론 (Conclusions)
- **이전**: 표지자의 신뢰성과 SSL 사전학습의 비효율성을 단정함.
- **Revision 2**: 탐색적 기술 가능성을 제시하되, 배경 간 질환 층화 분석 부재 및 다기관 외부 독립 검증 결여를 결론부에 직접 명시하여 과잉 해석을 방지함.

---

### 2.3 서론 (1. Introduction)
1. **과제 스펙트럼 재정의**:
   - 조기 암 감시(Surveillance) 대신 "B-mode 초음파에서 우연히 발견된 결절의 2진 감별 진단"으로 과제 범위를 엄격히 획정.
2. **비전형적 병변의 감별 난제 문헌 보강**:
   - 전형적 고에코 혈관종과 저에코 HCC 외에, 고에코 고분화 간세포암([6]) 및 저에코 테두리를 지닌 비전형 혈관종([7, 8])의 영상학적 모사(mimicking) 문제를 제시하여 연구의 임상적 배경 강화.
3. **Softmax 출력 한계 및 Similarity-based 추론의 필요성**:
   - Softmax의 과잉 확신(overconfidence)과 미보정 상태에서의 결정 강도(decision strength) 한계([11]) 제시.
   - 초음파 판독의가 수행하는 '전형적 질환 형태와의 기하학적 유사성 비교 추론'을 모사하기 위한 임베딩 공간 프로토타입 코사인 점수 제안 당위성 부여.

---

### 2.4 대상 및 방법 (2. Materials and Methods)

#### 2.1 연구 대상 및 데이터셋 구성
- **[핵심 삭제]**: 원저의 영상 선정 과정에서 *"확진 시점에 근접한 대표 B-mode 영상 1장이 선택되었다"*는 기존 문구를 **완전 삭제**함.
  - *사유*: SMC-LUD 논문 실사 결과 Clean subset은 744명 환자로부터 2,656장의 영상을 포함하므로 환자당 평균 3.57장이며, 1:1 매칭이 아니므로 기존 서술은 명백한 오류임.
- **[데이터 정합성 반영]**:
  - SMC-LUD 전체 코호트(1,021명, 5,385장: HCC 600명/2,716장, 혈관종 421명/2,669장) 명시.
  - Clean subset(744명, 2,656장: HCC 1,389장, 혈관종 1,267장) 명시.
  - 원저가 환자 단위(patient-level, 70:20:10)로 분할하여 동일 환자 영상이 train-val-test 세트 간에 교차되지 않음을 준용하였으나, 공개 데이터 파일명에 환자 식별자(patient ID)가 연동되어 있지 않아 본 연구팀이 직접 교차 검증은 못하였음을 투명하게 기술.

#### 2.2 모델 아키텍처 및 학습 프로토콜
- EfficientNetV2B0(7.1M) 기반 Hybrid Vision Transformer와 ResNet50V2(23.6M) 대조군 설정.
- 6개 조건(CE Only, CE+SupCon, NNCLR 미세조정)의 절제 연구(ablation study) 표준화.

#### 2.3 이중 출력 영상표지자의 정의
- Confidence Score: 소프트맥스 출력값 ($0 \sim 1$).
- HCC Cosine Score: **훈련 세트(Train set only, Data leakage 방지)** HCC 평균 벡터(Mean Prototype, $\mathbf{p}_{\text{HCC}}'$)와의 내적 ($[-1, 1]$).
- Hemangioma Cosine Score: 훈련 세트 혈관종 평균 벡터와의 내적.
- $\Delta\text{score}$: $\text{HCC Cosine} - \text{Hemangioma Cosine}$.
- k-means ($k=4$) 서브 프로토타입 기반 코사인 점수 산출을 민감도 분석용으로 병행 정의.

#### 2.4 임계값 설정 및 통계 분석 (전면 신설·보강)
- **독립 임계값 결정**: Validation 세트에서 Youden's J로 결정 후 Test 세트에 고정 적용.
- **신뢰구간 산출**: 모든 이진 분류 지표에 Clopper-Pearson Exact Binomial 95% CI 적용.
- **DeLong 검정의 한계 명시**: AUROC 1.000 천장값 조건에서 비열등성 단정 불가 명시.
- **[신규 통계 검정법 확립]**:
  1. **Rightward Shift (중심값 우측 이동)**: 검증·테스트 미학습 영상 대상 대응표본 **Wilcoxon signed-rank test** ($p < 0.001$) 및 paired t-test 적용.
  2. **Variance Reduction (분산 축소)**: 동일 영상 간 대응 분산(correlated paired variances) 차이를 검정하기 위해 **Pitman-Morgan test** ($p < 0.001$) 공식 도입.
  3. **검정 대상 표본**: True HCC ($n=417$; Val 277 + Test 140) 및 True Hemangioma ($n=381$; Val 253 + Test 128) 양방향 동시 적용.

---

### 2.5 결과 (3. Results)

#### 3.1 ~ 3.4 데이터셋 특성, Ablation, 테스트셋 성능, 이중 출력 비교
- **Table 1 & Table 1B**: 인구통계학적 특성 및 세트별 클래스 분포 제시.
- **Table 2 (검증 세트)**: 주력 모델(#4, EfficientNetV2B0 + CE+SupCon)의 성능 우위(AUROC 1.000, 7.1M) 및 NNCLR(#5, #6)의 성능 저하(AUROC 0.9986) 확인.
- **Table 3 (테스트 세트)**: 268장 중 FP=1(인덱스 576, 실제 혈관종), FN=0 보고.
- **Table 4 & 4B (출력 비교)**: 5가지 출력 형태 모두에서 동일한 1건만 오분류되어 cosine 지표의 단독 증분 이득(incremental benefit)을 입증하지 못했음을 정직하게 서술. ResNet50V2+NNCLR 조건에서 cosine AUROC가 0.0684로 붕괴(label-space inversion)된 원저 현상 각주 반영.

#### 3.5 미학습 영상 코사인 유사도 점수 분포 (SupCon 표상 프로빙 전면 개편)
기존의 단순 박스플롯(Fig 4, Fig 5) 비교를 전면 폐기하고, 검증·테스트 세트의 Unseen Case 전체($n=798$)를 대상으로 체계적 통계 수치 제시:

| 분석 대상군 | 백본 모델 | CE Only | CE + SupCon | 변화량 및 Wilcoxon 검정 | Pitman-Morgan 분산 축소 검정 | 연결 그림 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **True HCC**<br>($n=417$) | **EfficientNetV2B0** | $0.783 \pm 0.256$<br>(median $0.882$, $\operatorname{Var} 0.065$) | **$0.908 \pm 0.160$**<br>(median $0.970$, $\operatorname{Var} 0.026$) | **$+0.125$** ($95\%$ CI: $[0.110, 0.140]$)<br>$p < 0.001$ ($94.7\%$ 증례 상승) | **$-60.7\%$ 축소**<br>$t=16.84$, $p < 0.001$ | **Figure 4** |
| | **ResNet50V2** | $0.774 \pm 0.265$<br>(median $0.879$, $\operatorname{Var} 0.070$) | **$0.878 \pm 0.230$**<br>(median $0.974$, $\operatorname{Var} 0.053$) | **$+0.104$<br>$p < 0.001$ | **$-24.3\%$ 축소**<br>$t=4.06$, $p < 0.001$ | **Figure 4** |
| **True Hemangioma**<br>($n=381$) | **EfficientNetV2B0** | $0.937 \pm 0.050$<br>(median $0.955$, $\operatorname{Var} 0.00253$) | **$0.985 \pm 0.006$**<br>(median $0.985$, $\operatorname{Var} 0.00003$) | **$+0.048$** ($95\%$ CI: $[0.043, 0.052]$)<br>$p < 0.001$ ($t=19.71$) | **$-98.7\%$ 축소**<br>$t=106.23$, $p < 0.001$ | **Figure 5** |
| | **ResNet50V2** | $0.966 \pm 0.019$<br>(median $0.970$, $\operatorname{Var} 0.00036$) | **$0.990 \pm 0.004$**<br>(median $0.990$, $\operatorname{Var} 0.00002$) | **$+0.024$** ($95\%$ CI: $[0.022, 0.026]$)<br>$p < 0.001$ ($t=26.54$) | **$-95.6\%$ 축소**<br>$t=49.18$, $p < 0.001$ | **Figure 5** |

- **$\Delta\text{score}$ 역설 규명**: SupCon 적용 시 $\Delta\text{score}$ 중앙값이 $1.508 \rightarrow 1.076$으로 감소한 원인은 HCC뿐 아니라 혈관종 영상 역시 각자의 프로토타입($\approx 0.99$)으로 초구면 양 극단 수렴하면서 나타난 전체 기하 구조의 정규화 수축 현상임을 수학적으로 규명함.

---

### 2.6 고찰 (4. Discussion)

#### 4.2절: SupCon 정규화 방어 논거 신설 (Defense Statement)
- 소규모 초음파 영상 데이터에서 Vision Transformer는 귀납적 편향(inductive bias)의 결핍으로 인해 결정 경계를 단순 암기(shortcut memorization)하기 쉬움.
- AUROC 1.000 상태는 내부 표상의 일반화 여부를 분별하지 못함(Metric Saturation).
- CE 단독 모델은 미학습 데이터에서 분산이 크고 불안정했으나, CE+SupCon은 양 질환 모두에서 분산을 극단적으로 축소시키고($-60.7\%, -98.7\%$) 초구면 중심부로 강제 밀집시킴.
- 즉, **SupCon의 진정한 가치는 정확도 숫자가 아니라, 소규모 의료영상에서 ViT의 단순 암기를 방지하고 의미론적 표상을 정렬하는 필수 정규화기(essential regularizer)라는 점을 입증함.**

#### 4.3절: Softmax Calibration vs True Probability 문헌 통합
- Softmax 출력은 별도의 Calibration 없이는 실제 임상 유병 확률을 보장하지 않으며 Overconfidence를 유발함([11] Guo et al.).
- 의료영상 분야(흉부 X선, 안저 영상)의 Calibration Error 보고 문헌([16] Sambyal et al., [17] Rajaraman et al.)을 인용하여 본 연구 출력을 '상대적 결정 강도(Confidence Score)'로 제한 해석한 타당성을 논증함.

#### 4.4절: 엄격한 5대 제한점 (Limitations) 상세 개편
1. **스펙트럼 편향 및 선택 편향 (Spectrum & Selection Bias)**:
   - 과제가 감시 검사가 아닌 결절 감별에 국한됨.
   - HCC는 조직확진인 반면 혈관종은 영상진단이라는 참조표준 비대칭성 존재.
   - HCC 병변 크기 중앙값이 2.90 cm로 비교적 큼. (*"대표 1장 선별" 문구는 삭제*).
2. **배경 간 실질 정보 부재 (Absence of Background Liver Data)**:
   - 간경변, 만성 간염, 지방간 배경 정보가 없어 배경 음영을 단축 경로(shortcut)로 학습했을 가능성 배제 불가.
3. **외부 독립 검증 부재 (Lack of External Validation)**:
   - 단일기관 후향적 연구로, 타 기관, 타 장비, 타 검사자 환경에서의 일반화 성능 미확정.
4. **환자 대 영상 비(1:1 매칭 아님) 및 환자 내 상관성 (Intra-patient Correlation)**:
   - 744명 환자 대비 2,656장 영상(평균 3.57장/인)으로 1:1 대응이 아니며, 영상별 환자 식별자(patient ID)가 직접 연동되지 않아 환자 단위 bootstrap 분석 불가.
   - 영상 단위(image-level) exact CI 산출로 인해 환자 내 상관성에 의해 신뢰구간이 다소 좁게 추정되었을 가능성 명시.
5. **지표 천장 효과 및 추가 임상 증분 이득 미입증 (Ceiling Effect)**:
   - AUROC 1.000 도달로 DeLong 검정력 상실, Youden 임계값(0.0011) 왜곡, 테스트셋 오분류(FP=1)가 모든 출력에서 공통 발생하여 코사인 지표의 독자적 임상 구제 효과 미입증 인정.

---

### 2.7 참고문헌 (References)
- 기존 [1]~[15] 순차 번호 유지.
- EfficientNet([14]), DeLong([15]) 번호 정합성 확립.
- Discussion Calibration 논거에 맞추어 **[16] Sambyal et al. (2023)** 및 **[17] Rajaraman et al. (2022)** 신규 등재 완료.

---

## 3. 그림 및 표 개편 현황 (Figures & Tables)

### 표 (Tables)
- **Table 1**: SMC-LUD 전체 인구통계학적 특성 (연령, 성별, 종괴 크기).
- **Table 1B (신설)**: Clean subset 세트별(Train/Val/Test) 클래스별 영상 수 분포 ($n=2,656$).
- **Table 2**: 검증 세트 6개 조건 Ablation Study 결과 (Exact CI 및 파라미터 수).
- **Table 3**: 주력 모델의 독립 테스트 세트($n=268$) 성능 (오분류 1건 명시).
- **Table 4**: 이중 출력(Confidence vs Cosine Mean/k-means vs Delta Mean/k-means) 검증셋 컷오프 및 테스트셋 평가.
- **Table 4B (신설)**: 전체 6개 모델의 테스트셋 AUROC 비교 (NNCLR 라벨 공간 인버전 각주 명시).

### 그림 (Figures)
- **Figure 1**: t-SNE 임베딩 공간 시각화 (Mean vs k-means k=4 프로토타입 중심).
- **Figure 2**: 주력 모델 테스트 세트 3개 출력에 대한 혼동행렬 (TP 140, TN 127, FP 1, FN 0).
- **Figure 3**: Triple ROC Curves (ROC-A, ROC-B, ROC-C).
- **Figure 4 (전면 교체)**: `fig_hcc_cosine_distribution_comparison.png`
  - *구성*: True HCC($n=417$) 대상 4패널 (A: EfficientNet KDE/히스토그램, B: EfficientNet Paired Shift, C: ResNet KDE, D: ResNet Paired Shift).
  - *주석*: Wilcoxon $p < 0.001$, Pitman-Morgan 분산 축소 $-60.7\%$ ($p < 0.001$), 95% CI 명시.
- **Figure 5 (전면 교체)**: `fig_hemangioma_cosine_distribution_comparison.png`
  - *구성*: True Hemangioma($n=381$) 대상 4패널 (A: EfficientNet KDE/히스토그램, B: EfficientNet Paired Shift, C: ResNet KDE, D: ResNet Paired Shift).
  - *주석*: Wilcoxon $p < 0.001$, Pitman-Morgan 분산 축소 $-98.7\%$ ($p < 0.001$), 95% CI 명시. (기존 단순 박스플롯 완전 대체).

---

## 4. 생성 및 관리 대상 파일 일람

| 파일 경로 | 파일 형식 | 설명 |
| :--- | :---: | :--- |
| `paper_submission/revision2/manuscript_revised_marked.docx` | Word (.docx) | 변경 문장 RED, 기존 문장 BLACK 마킹된 최종 제출용 원고 |
| `paper_submission/revision2/manuscript_revised_final.txt` | Text (.txt) | 최종 개편 전문 텍스트 (Markdown 헤더 포함) |
| `paper_submission/revision2/tables_and_figures_revised.docx` | Word (.docx) | Table 1~4B 및 Figure 1~5 최종 캡션이 포함된 제출용 문서 |
| `paper_submission/revision2/fig_hcc_cosine_distribution_comparison.png` | Image (.png) | Figure 4 공식 이미지 (300 DPI, 4-Panel HCC Cosine Shift) |
| `paper_submission/revision2/fig_hemangioma_cosine_distribution_comparison.png` | Image (.png) | Figure 5 공식 이미지 (300 DPI, 4-Panel Hemangioma Cosine Shift) |
| `paper_submission/revision2/supcon_probing_defense_plan.md` | Markdown (.md) | Reviewer A 반박용 표상 프로빙 방어 전략 기획서 |
| `paper_submission/revision2/comprehensive_revision_history.md` | Markdown (.md) | **(본 문서)** 리비전 이력 종합 보고서 |
| `paper_submission/revision2/generate_revised_manuscript.py` | Python (.py) | 원고 텍스트 및 marked.docx 빌드 자동화 스크립트 |
| `paper_submission/revision2/generate_revised_tables.py` | Python (.py) | 테이블 및 그림 캡션 docx 빌드 자동화 스크립트 |
