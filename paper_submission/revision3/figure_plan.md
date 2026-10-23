# 그림 점검 계획 (테스트 세트 전용 · 운영점 이원화 반영)

대상: `paper_submission/revision2/` 의 그림 8개 (Fig 1, 2-A/B/C, 3-A/B, 4, 5)
원칙: ROC·컷오프·혼동행렬은 테스트 세트 기준으로만 보고, 확률 0.50 은 주 분석, 닮음 지표는 탐색적.

## 1. 그대로 쓰는 그림

| 그림 | 파일 | 이유 |
|---|---|---|
| Fig 1 | `fig 1 TSNE visualization.jpg` | 임베딩 시각화. 컷오프·ROC 와 무관. |
| Fig 2-B | `fig 2-B Confusion matrix-B.jpg` | HCC cosine, Youden 운영점. 수치 그대로(140/1/0/127). 캡션에 "탐색적"만 표기. |
| Fig 2-C | `fig 2-C Confusion matrix-C.jpg` | Δscore, Youden 운영점. 수치 그대로. |
| Fig 3-B | `fig 3-B.jpg` | 테스트 triple ROC. 세 곡선 AUROC 1.000. |
| Fig 4 | `fig_4_hcc_cosine_distribution_comparison.png` | 분포 비교. 컷오프 기반 아님(Val+Test 합산 명시 필요). |
| Fig 5 | `fig_5_hemangioma_cosine_distribution_comparison.png` | 위와 같음. 단 아래 4번 명칭 문제 확인 필요. |

## 2. 다시 그린 그림 (완료)

- **Fig 2-A (Confusion matrix-A)**: 지금은 옛 컷오프 0.001 기준(140/0/1/127).
  → 0.50 기준으로 재출력(134/0/6/128), 제목 "Test, cutoff = 0.50".
- **Fig 2-B/2-C (선택)**: 수치는 그대로 두고, 축·제목 스타일만 2-A 와 맞추려면 같은 스크립트로 재출력.

## 3. 뺀 그림 (완료)

- **Fig 3-A (Triple ROC, Val)**: 테스트 전용 보고 원칙과 어긋남. 본문 Figure 3 은 3-B 단독으로 구성(이미 반영).

## 4. 더할 그림 (보류 — 이번 개정에서는 넣지 않음)

1. **운영점 실패 지도 1장** — 테스트 268장의 confidence 분포 위에 0.50 선을 긋고, 놓친 HCC 6장(0.03–0.40)과 위양성 혈관종 1장(0.0014)을 표시. 슬라이드 10 의 메시지를 그림 한 장으로 옮긴 것.
2. **등가 구간 띠** — 지표별 임계값 축에 등가 구간을 밴드로 표시. "컷오프는 점이 아니다"를 시각화.
3. McNemar 2×2 는 그림보다 표가 적절하므로 그림 추가는 하지 않음.

## 5. 정리해야 할 불일치

- `fig_4_...` 는 density + paired 4패널인데 원고 Figure 4 캡션은 "boxplot"으로 기술 → 캡션 수정 또는 boxplot 재출력 중 택1.
- `fig_5_...` 는 **hemangioma cosine** 분포인데 원고 Figure 5 는 **Δscore** boxplot → 파일 교체 또는 캡션 수정 필요.
- Fig 2 캡션에 패널별 운영점 표기(2-A 0.50 / 2-B·2-C Youden·탐색적).

## 6. 처리 결과 (2026-10-23)

- Fig 2-A: `figures/make_confusion_matrix_a.py` 로 0.50 기준 재출력(134/0/6/128). 완료.
- Fig 3-A(검증 ROC): 전달 세트에서 제외(테스트 전용). 완료.
- Fig 4/5: 그림은 그대로 두고 **캡션을 그림에 맞춤**(HCC n=417 / hemangioma n=381, density + paired shift). 완료.
- Fig 2 캡션: 패널별 운영점 표기(2-A 0.50 / 2-B·2-C Youden·탐색적). 완료.
- Fig 3 캡션: 테스트 전용, DeLong 무정보, 비열등 주장 없음 명시. 완료.
- 4-1·4-2(새 그림): 넣지 않음.

전달 세트: `revision3/figures/` (Fig 1, 2-A/B/C, 3, 4, 5).
