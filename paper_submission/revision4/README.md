# revision4 작업 상태 — 2차 수정(KJFP-26-0060, Minor Revision)

편집위원회 2차 결정: Minor Revision, 마감 2026-12-25(60일). 이번 라운드 수정 표시는 **파란색**.

## 끝난 것 (지시받은 3건)

### 1. 참고문헌 확보·검증 → `REFERENCE_AUDIT.md` + `CITATION_SUPPORT_AUDIT.md`
- **인용-문장 대응 전수 점검(14개 인용 지점)** 완료 → `CITATION_SUPPORT_AUDIT.md`. HIGH 2건·MED 1건·부분 2건 확인 후 **전부 반영**:
  - ref 6(Lin 2017)이 반대 방향 문장에 붙어 있던 문제 → 혈관종 문장으로 이동, 고에코 HCC 문장에는 **Rapaccini 2004 Liver Int**(2 cm 미만 HCC 17.0% 고에코) 신규 배치
  - "저에코 테두리" 서술을 Moody 1993(고에코 테두리)과 맞게 **"내부 에코가 부분적으로 저에코이거나 테두리 소견"** 으로 수정
  - 근거 없던 "기존 연구는 단일 softmax만 출력" 주장 → 범위 한정 + 예시 인용([10,11])
  - Sirli(CEUS 논문)를 난제 문장에서 CEUS 문장으로 이동
  - **최종 18개**로 Vancouver 순서 재배열(파란색 표시 유지)
- ref 6: 저자 오류 수정(Chou CT… → **Lin S, Zhang L, Li M, et al.**, DOI 10.1097/MD.0000000000009192)
- ref 7: 문장 미지지 → **Moody AR, Wilson SR. Radiology. 1993;188(2):413-417** (B-mode 형태학)
- ref 8: CT/MRI LI-RADS는 무관 → **Choi JC, et al. Gut Liver. 2009;3(3):226-230** (저에코 테두리)
- ref 17: **잘못된 서지** 발견·수정 → `PLoS One. 2022;17(1):e0262838`(기재값 e0278455는 각막원추증 논문)
- 17건 전수 Crossref/PubMed 대조 완료. 남은 확인 1건: ref 13(SMC-LUD) `13:649` 문서번호 → 게재본 확인 필요.

### 2. Figure·Table 통합
- **Figure 2 = 1장 3패널**로 통합: `figures/fig 2 combined confusion matrices.png`
  (A confidence@0.50, B HCC cosine@Youden, C Δscore@Youden, 모두 n=268, 수치는 표와 일치)
  생성기 `figures/make_figure2_combined.py` — 원자료 `cosine_probe_result/raw_cosprobe_effnet_supcon.csv`에서 재계산.
- **Table 4 = 통합표 1개**: 패널 A(주력 모델 운영점 성능, 5행) + 패널 B(6개 모델 테스트 AUROC, 7행).
  기존 Table 4·4B를 하나로 합치고, 각주를 1개로 통합. 생성기 `generate_tables_v4.py`.
- 본문 인용 갱신: `Table 4A`(운영점), `Table 4B`(모델 간 AUROC), `Figure 2A–2C`.

### 3. Supplementary Methods 분리 → `Supplementary_Methods_v1.docx` / `.txt`
- **S1** 모델 아키텍처(패치 144+CLS 1, d=128, 4블록, 헤드 8, MLP 384, dropout 0.1, 헤드 128)
- **S2** 학습 프로토콜(384², 배치 16, 100 epoch, AdamW clip 1.0+EMA, CosineDecay 1e-6→5e-5/warmup 10, 증강, seed 42)
- **S3** 손실함수·비교 조건(CE / CE+SupCon 0.5×·temp 0.07 / NNCLR 1단계 temp 0.1·queue 8192·proj 256)
- **S4** 프로토타입·유사도 산출(mean prototype, k-means k=4 평균 코사인, 훈련 세트 한정)
- **S5** 임계값·등가 구간·통계 절차(Wilson, DeLong, McNemar exact, paired t/Wilcoxon/Pitman-Morgan)
- **S6** 재현 환경(t-SNE perplexity 30, 시드 42)

값은 전부 학습 코드에서 읽었다(`classification_with_supcon.ipynb`, `full_training_notebook.ipynb`, `NNCLR_ConvHybrid_run.ipynb`). 코드로 확인하지 못한 항목은 채우지 않았다.

## 본문에서 함께 손본 것
- Methods 2.2 → 설계 중심 1문단으로 축약 + "S1–S3 참조" (심사자 comment 5)
- Methods 2.4 등가 구간 절차 설명 → S5로 이동 (본문 1문장으로)

## v5 — KJFP 양식 적용 (2026-10-27)

`KJFP_STYLE_CHECKLIST.md`(SSOT) 기준으로 서식·분량·구조를 재구성했다.

- `manuscript_v5_kjfp.docx` — A4·3 cm·두 줄·바탕체 10 pt·쪽번호, 위첨자 인용, 영문초록 243 words,
  국문요약 791자, 본문 9,993자, 참고문헌 18개(6인 규칙·`저널 연도; 권: 면` 형식)
- `Title_Page_v1.docx` — 저자 김현수(0009-0009-8896-5344)·이진희*(0000-0001-7762-4952), Presbyterian Medical Center(예수병원) 전주 54987, 교신저자 이진희 jin56878@jesushospital.com, 원고 연락 김현수 kimhs950627@gmail.com, 연구비·이해관계 없음 〔전화/팩스·상세주소 확인〕
- `tables_v5.docx` — Table 3·3B 통합, 전 표 3선형, 주석 위첨자 a·b·c
- `figures_kjfp/` — JPEG 600 dpi (Fig 2·3은 원자료에서 실제 600 dpi 재렌더)
- `verify_kjfp.py` — 기계 검증 37항목 전부 PASS
- 생성기: `build_v5_source.py` → `manuscript_v5_source.md` → `generate_manuscript_v5.py`

- `Response_to_Reviewers_v6.docx` — 편집위 1–7 + Reviewer A/B 대응, 모든 항목에 실제 수정 위치 명시
- `generate_response_v6.py` — 답변서 생성기(문구·수치 상호검증 포함)

## Commentary 대응 완료 (2026-10-27 2차)

- Comment 1: 서론 4문단 재작성 — 판독자 76.1/76.9% vs 모델 86.5/85.5%(AUC 0.924, 정확도 84.7%) 근거 추가, 전형성 논리, `결론적으로 … McNemar P = 0.125`
- Comment 2: 고찰 3문단 신설 — 지표 분해, 임상 사용 4상황, 읽기 한계, 적용 범위
- Comment 5: Methods 2.2에 학습방식 목적 3줄 + NNCLR 가설 명시, 손실식·코사인 공식 → Supplementary S3/S4
- 감축: Results 3.2, 고찰 첫 문단, 제한점 둘째·셋째 압축 + k-means → S4

## 아직 안 한 것 (지시 대기)
- **용어 표기 일관성**(comment 3) — 국·영문 병기 정리, 기술용어 의미 설명
- **Discussion 2번 보강**(comment 2) — similarity score가 도움이 되는 임상 상황과 해석 한계
- Response to Reviewers v6, Cover letter 갱신, blue 표시 최종 렌더 확인(Word 없음 → 육안 확인 필요)

## 파일
- `manuscript_v4_marked.docx` (파란색 표시, 내부 확인용) · `manuscript_v4_clean.docx` · `manuscript_v4_final.txt`
- `tables_v4.docx` (통합 Table 4 + 나머지 표 + 그림 캡션)
- `Supplementary_Methods_v1.docx` / `.txt`
- `figures/fig 2 combined confusion matrices.png` + 생성기
- `REFERENCE_AUDIT.md`
