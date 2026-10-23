# 비주얼 디렉션 — LiverDx

## 톤
**"임상용 계측기(instrument)"**. 화려한 의료 앱이 아니라 초음파 장비 콘솔의 정밀함.
다크 슬레이트 배경 + 값이 도드라지는 큰 숫자 + 얇은 1px 헤어라인. 장식 없음.

- 참조 감각: 초음파 모니터, Fluke 계측기, Bloomberg 터미널(정보 밀도).
- 피할 것: 보라-인디고 그라디언트, 과한 라운드(24dp+), 그림자 카드 남발, 균등 배분 팔레트, Inter/Roboto 기본 톤.

## 팔레트 (지배색 1 + 날카로운 악센트)
| 토큰 | 값 | 용도 |
|---|---|---|
| `Ink` | `#070A0F` | 앱 배경 (window background) |
| `Surface` | `#0F151D` | 카드/슬롯 배경 |
| `SurfaceHigh` | `#17202B` | 눌림·선택 상태 |
| `Hairline` | `#243141` | 1px 구분선/테두리 |
| `TextPrimary` | `#EAF1F8` | 값·제목 |
| `TextSecondary` | `#93A6B8` | 라벨 |
| `TextMuted` | `#5E7386` | 면책·보조 |
| `Malignant` | `#FF3D5A` | HCC / 위험 / 상승 |
| `Benign` | `#22D3A6` | Hemangioma / 양성 |
| `Caution` | `#F5B544` | 컷오프 마커 / 경계 |

정리 패널의 glare 경고색은 `Caution`, fp16 경량 모델 배지도 `Caution` (fp32 는 `TextMuted`).
컬러맵: attention rollout = **inferno 계열**(검정→보라→주황→흰색 아님, 흑→자주→주황),
프로토타입 유사도 = **turbo 계열**(파랑→초록→노랑→빨강). 두 맵을 색으로 구분해 혼동을 막는다.

## 타이포
- 디스플레이/숫자: **Space Grotesk** (OFL, 자체 번들). 숫자에 tabular 특성이 좋고 계측기 느낌.
  점수는 `fontFeatureSettings = "tnum"`, 크게(28–44sp), `FontWeight.Bold`.
- 본문/라벨: 동일 패밀리 400/500. 별도 본문 폰트를 추가하지 않는다 (한글은 시스템 폰트 폴백).
- 라벨은 11sp + `letterSpacing 0.1em` + 대문자 → 계기판 눈금 라벨 (`AppType.gaugeLabel`).

## 레이아웃
- 16dp 기본 거터, 카드는 기본 라운드 6dp(작게), 헤어라인 1dp `Hairline`.
- Verdict 카드는 좌측 4dp 컬러 바 + 큰 퍼센트 우측 정렬. 카드 그림자 대신 1px 테두리.
- 점수 4행은 **계기판 행**: 라벨(좌) — 값(우, 모노 숫자) — 컷오프 미니바(아래 2dp).
  컷오프를 넘으면 값이 `Malignant`, 아니면 `Benign`, 컷오프 마커는 `Caution` 눈금으로 표시.
- 사진 정리 패널은 점수판과 같은 카드 문법: 헤더 라벨 + 계기 행(라벨/영문 id/Switch) + 헤어라인 +
  리샘플 칩 2개. Switch 만 Material 기본 형태를 쓰고 색만 토큰으로 맞춘다(접근성 우선).
- 카메라 화면은 배경 위 전체 프리뷰 + 72 % 폭 정사각 가이드(1px Benign 테두리) + 하단 3버튼.
- 촬영 버튼만 채움(Benign), 나머지는 Surface+헤어라인. 라이트 토글은 상태를 글자로 표시한다.
- 시각화 3 패널은 1:1 정사각 프레임 3개 균등 배치(`ContentScale.Fit` — 맵의 가장자리를 자르지
  않기 위해), 위에 대문자 라벨·아래 소문자 캡션, 컬러맵 패널에만 램프 바.
- 배경 깊이: 단색 대신 Ink 위에 아주 옅은 radial 그라디언트(중심 #0F151D → 가장자리 #070A0F).

## 모션
- 결과 도착 시 **한 번의 스태거드 리빌**: Verdict(0 ms) → 점수행(40 ms 간격) → 패널(200 ms).
  각 항목 `fadeIn + slideInVertically(8dp)`, 220 ms, `FastOutSlowInEasing`.
- 값 카운트업은 하지 않는다 (판독값은 즉시 확정되어야 함).
- `prefers-reduced-motion` 대응: Android `Settings.Global.ANIMATOR_DURATION_SCALE` 가 0이면
  애니메이션을 건너뛰고 즉시 표시. (Compose 는 시스템 스케일을 따름 — 별도 분기 불필요하나 문서화)

## 금지 목록 (AI 티 방지)
1. 보라/인디고 그라디언트 — 없음.
2. 전체를 감싸는 큰 라운드 카드 안의 또 다른 카드 — 없음 (2단 중첩 금지).
3. 의미 없는 아이콘 나열 — 아이콘은 촬영/갤러리 2개만.
4. 이모지 없음.
5. "혁신적인", "스마트한" 류 카피 없음 — 값과 단위만.
6. 그림자 4dp 이상 없음 — 1px 테두리로 대체.
