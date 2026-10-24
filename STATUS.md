# STATUS — 2026-10-24

## 지금 어디까지 왔나

- **2026-10-24 대한가정의학회(KFM) 추계학술대회 발표 완료** — 8분 발표 + 2분 질의, 사전 리허설·큐카드 사용.
- 발표 패키지와 논문 revision3 패키지는 모두 커밋·푸시됨 (`e92a5dd` 기준).
- 다음 작업은 **논문 재제출**이며, 아직 제출하지 않았다(준비만 된 상태).

## 발표 산출물 (재사용 위치)

- 덱: `presentation/hcc_dual_output_talk_v4.html` / `.pdf` — **13장**
  - 웹 게시본: https://kimhs950627.github.io/hcc-vs-hemangioma/talk/
  - 7장 = 성능 표(컷오프 행 포함), 8장 = "컷오프는 점이 아니라 구간이다"(클릭 시 확대)
- 대본: `presentation/발표대본_v5.md` / `.pdf` — 13절, 말하는 시간 7분 35초 + 전환 25초
- 큐카드: `presentation/cuecards/큐카드_발표.pdf` (10장/19카드), `큐카드_토론QA.pdf` (8장/16카드)
- 8장 그림: `presentation/slide8_figure.svg` + 생성기 `slide_explainer_svg.py` (원자료에서 재생성 가능)
- 백업 설명판: `presentation/figures_youden_explainer.png` (1920×1080, 질의용)

## 논문 패키지 (`paper_submission/revision3/`)

- `manuscript_v3_marked.docx` (표 없음, 수정 부분 빨간색) / `manuscript_v3_final.txt`
- `tables_v3.docx` (표 7개 + 그림 캡션), `figures_v3.docx` (그림 5개, 이미지 7개)
- `Response_to_Reviewers_v5.docx` / `.txt` (v4 레이아웃 유지)
- `figures/` 배포 세트, `captions_v3.py`(캡션 단일 출처), `equivalence_interval.py`(구간 재현)

## 확정된 수치·정의 (재제출 때 그대로 쓴다)

- 주 분석: confidence **0.50**, **테스트만** 보고 → 134/0/6/128 (민감도 95.71 · 특이도 100 · 정확도 97.76)
- 닮음 지표(탐색적, 검증 유도): 140/1/0/127 (100 · 99.22 · 99.63)
- 불일치 6 대 1, McNemar exact **p = 0.125** → 우월·비열등·개선 주장 없음
- 컷오프 = 검증 Youden 구간의 **위쪽 끝** = 검증에서 가장 낮은 점수를 보인 HCC 증례의 값
- 등가 구간(검증): confidence (0.4498, 0.5400] · cosine (−0.0498, −0.0048] · Δscore (−1.0001, −0.9585]
- Δscore는 **소개 수준**으로만 언급 (추가 분석 보류 상태)

## 남은 리스크 / 확인 필요

1. **Word 렌더 최종 확인** — 이 환경에는 LibreOffice가 없어 docx를 눈으로 확인하지 못했다. `manuscript_v3_marked.docx`, `tables_v3.docx`, `figures_v3.docx`를 Word에서 한 번 열어 표 폭·페이지 나눔·빨간 표시를 확인해야 한다.
2. **작업 트리 119개 파일이 dirty** — Windows git(`core.autocrlf=true`)와 LFS 포인터 차이로 보인다(줄바꿈만 다른 파일 97개 + 바이너리/포인터 22개). 내용 커밋 금지. 필요하면 `git lfs status`로 확인 후 별도로 정리한다.
3. **gh CLI 미설치** — GitHub 조회/이슈 관리는 REST 토큰(`~/.config/gh/hosts.yml`)으로만 가능.

## 다음 한 걸음

`paper_submission/revision3/RESUBMISSION_CHECKLIST.md` 순서대로 진행한다. 첫 항목은 저널·투고 시스템 확정이고, 그 다음이 Word 렌더 확인 → 수치 감사 → 파일 세트 고정이다.
