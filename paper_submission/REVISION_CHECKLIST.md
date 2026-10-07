# HCC vs hemangioma: revision 실행 지침

상태: 계획/검증 대기. 이 문서는 확정된 결과 보고서가 아니다. 연구자와 LLM agent는 체크박스를 증거 확보 뒤에만 완료 처리한다.

## 0. 비협상 원칙

- 연구 질문: 발견된 간 국소 병변의 B-mode 초음파에서 HCC와 혈관종의 **감별진단 보조**. 감시·조기 발견·선별검사의 민감도 또는 무병변군 검출 성능이라고 주장하지 않는다.
- 원자료 SMC-LUD 전체 1,021명/5,385장과 본 연구 Clean subset 744명/2,656장을 구분한다. Table 1 전체 환자 특성을 Clean 특성으로 오인하지 않는다. 원저와 실제 자료에서 각각 재확인한 수치만 쓴다.
- 원저는 patient-level split을 보고한다. 본 연구에서 **원저 제공 분할을 실제 그대로 사용했는지** 실행 기록/코드로 확인한 뒤 인용한다. 분석 메타데이터에 patient ID가 없으므로 실제 사용 영상에서 환자 ID 교집합 0을 독립 검증했다고 주장하지 않는다. 파일명/폴더명에 익명 ID가 있으면 확보 가능성을 별도 점검한다.
- patient-level split ≠ patient-level metric ≠ patient-cluster bootstrap. 영상 score를 영상 단위로 재표본추출하면 image-level CI임. 환자 매핑 없이 환자 단위 집계·cluster CI를 생성하거나 그렇게 이름 붙이지 않는다.
- 숫자·임계값·표·그림은 원 예측값과 재현 코드로 검증하고, 검증 전의 1.000 AUROC 및 성능 주장을 확정 결과로 재사용하지 않는다. 저널 답변서에는 수행 사항과 수행 불가 사항을 구분한다.
- 외부 검증은 계획됨/진행 중임. 실제 결과가 나오기 전 완료되었다고 쓰지 않는다. 환자 식별정보·원본 임상자료·토큰 등 비밀은 저장소에 올리지 않는다.

## 1. 근거 및 저장소 지도

- 원 논문: Tak J et al. *Scientific Data* (2026), DOI 10.1038/s41597-026-07023-7; https://www.nature.com/articles/s41597-026-07023-7 . 원저의 환자별 그룹화·환자 단위 분할·혈관종 참조표준을 원문에서 인용하고 원고가 이용한 split과 동일한지 확인한다.
- 원고 기준: `paper_submission/manu_before_revision.docx` 및 `paper_submission/manuscript_revised.docx`/`.txt`. 후자는 이전 형식 수정본이지 이번 심사 의견을 반영한 최종본이 아니다. 최종 정본 하나를 선언하고 변경 이력을 기록한다.
- 표/그림: `paper_submission/table 1.docx`, `table 1b.docx`, `table 3.docx`, `table 4.docx`, `table 4b.docx`, `fig 2-*.png`, `fig 4.png`, `fig 5.png`.
- 분석: `dataloader.py`, `validation/`, `cosine_probe_result/raw_cosprobe_*.csv`, `experiment_registry/`, `ambiguous_case_analysis.py`, `misclassified_rescue_analysis.py`. CSV 열과 환자 ID 보유 여부는 아직 별도 확인 필요. `validation/extval.py` 파일의 존재는 외부 검증 완료의 증거가 아니다.
- 발표: `lecture/` 및 `poster.html`. 원고 결과가 확정된 후 갱신한다.
- `paper_submission/revision_script.py`에 '동등' 및 'SupCon 필수'라는 구식 초록 문구가 하드코딩되어 있음. 스크립트를 먼저 수정하지 않으면 과장된 문장이 재생성됨.

## 2. 분석 게이트 및 체크리스트

### A. 코호트·분할 확인 [P0]

- [ ] SMC-LUD 원저의 전체/clean/클래스별 환자·영상 수, 진단 근거, 제외기준을 대조한 source-of-truth 표를 만든다.
- [ ] 실제 실험이 원저 제공 train/val/test split을 그대로 썼는지 dataloader와 실행 기록으로 확인한다. 원저와 달리 재분할했다면 표현을 바로잡는다.
- [ ] train 1,858 / val 530 / test 268장의 파일 목록·클래스 수를 재현한다. 환자 매핑이 없으면 교집합 검증 불가 및 근접 중복 프레임 배제 불확실성을 적는다. 식별 가능한 익명 폴더가 있다면 중복 검사와 로그를 남긴다.
- [ ] Clean subset 연령/성별, HCC·혈관종 크기, HBV/간경변/지방간 정보를 **실제로 확보할 수 있는지** 확인한다. 없는 변수는 전체집단 Table 1에서 옮기거나 추정하지 않는다.

### B. 라벨·점수·성능 재현 [P0]

- [ ] 양성 클래스를 HCC로 고정하고 원 예측값에서 split별 TP/FP/FN/TN, 분모, threshold, sensitivity/specificity/PPV/NPV/F1/accuracy/AUROC를 재계산한다.
- [ ] Val: TP 276/FP 0/FN 1/TN 253 및 정확도 99.81%와 Table 3의 TP 277/FN 0/100% 충돌을 해결한다. Test: FN 0/FP 1이라는 Table 3·최종 Figure 2 범례와 Results 3.3의 반대 기술을 대조한다. 추측으로 숫자를 바꾸지 않는다.
- [ ] Results 3.5의 'true HCC n=381'을 원자료에서 검증한다. Val+Test HCC는 원고 기준 277+140=417장, 혈관종은 253+128=381장임. Figure 4/5 생성 코드의 label mapping, 필터, 행 수, score 분포·threshold와 Table 4의 오류 수를 대조한다. 분석 자체가 틀렸으면 그림과 통계를 재생성한다.
- [ ] Methods 2.4의 mean prototype 1개 정의와 k-means 4개 정의를 분리한다. 어떤 score/prototype/임베딩 계층을 Table 4·각 그림·외부 검증에 사용했는지 고정하고, prototype은 훈련 세트만으로 산출했는지 확인한다.
- [ ] 운영 임계값이 validation에서만 정해지고 internal test/external에 고정됐는지 코드로 확인한다. 단위·방향(HCC 점수가 클수록 양성)을 기록한다.

### C. 추정·해석 [P1]

- [ ] 영상별 score에 대한 image-level bootstrap 95% CI를 산출하되 이로 환자 내 상관성을 교정했다고 주장하지 않는다. 같은 환자 영상이 상관될 수 있어 CI가 좁아질 위험을 명시한다.
- [ ] 환자 ID가 추후 확보되면 환자/병변별 사전 지정 score 집계(예: 평균 또는 최대값) 및 환자 단위 cluster bootstrap을 추가하고 양자를 명확히 구분한다.
- [ ] DeLong p>0.05 또는 AUROC 1.000을 '비열등/동등' 근거로 쓰지 않는다. 비열등성 마진·설계를 사전 규정하지 않았다면 관련 주장을 삭제한다.
- [ ] CE-only의 HCC cosine/Δscore도 AUROC 1.000임을 반영하여 'SupCon 필수' 삭제. 동일 영상의 CE-only vs SupCon score 비교는 paired 분석으로, 영상 간 상관성이 있으면 그 한계도 표시한다. 점수 분포 상승 ≠ 감별 성능/임상효용 개선.
- [ ] Δscore가 confidence에 추가하는 증분 이득을 동일 사례·고정 운영점에서 평가한다. 이득을 확인 못 하면 독립적/보완적 임상 가치 주장 삭제.
- [ ] Results 3.6의 confidence 0.4–0.6 '구제'는 운영 cutoff 0.0011상 전부 HCC 판정이고 CE-only 7건과 SupCon 5건도 서로 다른 사례군임. 해당 분석/Discussion 문구를 삭제하거나 동일 사례·진짜 오분류에 대한 사전 명세된 분석으로 교체한다.

### D. 편향·설명가능성 [P1]

- [ ] 혈관종은 영상 소견/판독 기준, HCC는 병리 기준이라는 원저 참조표준 차이를 Methods에 정확히 쓰고 Discussion에 spectrum/selection bias를 밝힌다. 원저에 없는 일괄 CT/MRI/CEUS 추적 조건을 만들어 쓰지 않는다.
- [ ] 배경 간(HBV, 간경변, 지방간) 층화 자료가 있으면 분석한다. 없다면 모델의 배경 간 shortcut 가능성과 실제 감시 집단에 대한 미평가를 핵심 제한점으로 적는다.
- [ ] 가능하면 병변/주변 간을 함께 보여주는 Grad-CAM/attention map을 오류 사례 중심으로 제시한다. heatmap 자체를 shortcut 부재의 증거로 해석하지 않는다.

### E. 소규모 외부 검증 [P1, 결과 대기]

- [ ] 내부와 독립적인 외부 기관/자료 출처, 시기, 검사자·장비, 선정/제외, 혈관종과 HCC 참조표준, 환자/병변/영상 수, 원자료와 중복 방식을 기록한다.
- [ ] 외부 데이터를 보기 전에 최종 모델 가중치·전처리·prototype·score 산식·내부 validation에서 고정한 threshold·평가 단위 및 분석 코드를 동결한다. 외부 자료에서 threshold를 재최적화했다면 별도 조정 결과로 표시하며 이를 독립 외부 검증이라 부르지 않는다.
- [ ] 외부 TP/FP/FN/TN, AUROC, 민감도, 특이도와 95% CI를 보고하고 소표본으로 CI가 넓어짐을 설명한다. 환자 ID가 외부에는 있으면 환자 단위 집계 및 cluster CI를 따로 제시한다. 내부의 image-level 결과와 외부 patient-level 결과를 동일 단위처럼 비교하지 않는다.
- [ ] 외부 성능이 낮으면 숨기지 않고 domain shift, 사례 스펙트럼, 진단 기준 차이와 함께 서술한다. 단일 소규모 외부 세트는 광범위한 임상 적용의 입증이 아님.

## 3. 원고 위치별 수정 지도

| 위치 | 조치 |
|---|---|
| 제목·국/영문 초록 | 감시·조기 발견 중심 문장을 발견 병변의 HCC–혈관종 감별 보조로 바꿈. 확정 숫자와 외부 검증 결과만 반영. '동등', 'SupCon 필수' 삭제. |
| Introduction | 불필요한 1.1/1.2 소제목 삭제. 감시 민감도 47%를 본 연구의 성능 비교 근거로 사용하지 않음. 환자 스펙트럼과 진단 과제 한정. LI-RADS와 직접 동등하다는 식의 비약 제거. |
| Methods 2.1 및 Table 1/1B | 원자료/clean subset 분모 분리, 영상 제외 및 혈관종 진단 근거, 원저 인용 환자 단위 split과 현 연구의 독립 검증 불가를 구분. Clean 변수만 Clean 표에 기재. |
| Methods 2.4–2.6 | mean vs k-means prototype 정의, score 산식 및 사용 지점 통일; threshold 결정 시점, 평가 단위와 image-level CI를 기술. DeLong 비열등 해석과 독립 t-test 수정. 외부 검증 프로토콜·참조표준·동결 정책 추가. |
| Results 3.1–3.5, Tables 2–4B, Figures 1–5 | 원 자료에서 수치 재생성. Figure 1 split 표기, Table 3 confusion matrix, 3.3 FP/FN, 3.5 n=381 라벨/그림 분포, Figure 4/5 통계/threshold 일치 확인. Results의 Methods 반복 삭제. |
| Results 3.6 | 구제 효과 삭제가 기본. 유지하려면 동일 사례·실제 운영 임계값·증분 이득으로 재분석한 후만 허용. |
| Discussion/Conclusion | HCC 감시 및 일반화 주장 축소. 참조표준 차이, Clean 선택 편향, 배경 간 교란, 환자 매핑 부재·영상 단위 CI의 한계, 외부 소규모 검증의 범위를 서술. SupCon/NNCLR 인과 해석 축소. |
| 표/그림·발표자료 `lecture/` | 제목과 별표 설명을 각 표/그림의 caption/footnote로 이동. 원고가 확정된 다음 발표자료 숫자·제목·한계를 동기화. |

## 4. LLM agent 핸드오프/완료 기준

1. 데이터·원저 근거와 분석 재현 로그를 먼저 수집한다. 원본 숫자/그림은 검증 전 수정하지 말고 불일치 목록에 올린다.
2. 내부 오류 해결 → 외부 검증 사전 고정 및 실행 → 통계·표·그림 재생성 → 국/영문 원고 → 심사 답변서 → 발표자료 순서로 작업한다.
3. 각 변경에 `근거(원저/코드/예측 CSV) | 이전 문장/숫자 | 새 문장/숫자 | 재현 방법 | 검증 상태`를 기록한다. 데이터 부재는 `NOT AVAILABLE`, 미실행은 `PENDING`, 실패는 `FAILED`로 구분하고 값·p-value·CI를 창작하지 않는다.
4. 최종 QA: 모든 표·그림·본문·초록의 분모/양성클래스/평가단위/threshold/CI/외부 자료 범위가 일치하는지 점검한다. 구식 초록 문구를 생성하는 `revision_script.py`도 최종 문안과 동기화한다.

근거: Tak et al. SMC-LUD DOI 10.1038/s41597-026-07023-7; `paper_submission/manu_before_revision.docx`; `paper_submission/manuscript_revised.docx`. 이 문서의 미완료 체크박스는 실제 수행 결과가 아님.
