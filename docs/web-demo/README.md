# Web demo — QR 로 여는 브라우저 내 추론

학회장에서 QR 을 띄워 "여기서 추론을 해볼 수 있습니다" 로 쓰기 위한 정적 페이지임. 방문자는
**설치·로그인 없이** 자기 폰 브라우저에서 바로 돌릴 수 있고, 아이폰도 됨.

```
QR → https://kimhs950627.github.io/hcc-vs-hemangioma/web-demo/
```

사진은 **기기 밖으로 나가지 않음**(모델 파일만 내려받고, 추론은 방문자 기기에서 실행).

## 무엇이 들어있나

| 파일 | 내용 |
|---|---|
| `index.html` | 페이지 + 스타일 + 스크립트 (외부 의존은 아래 vendor 뿐) |
| `model/hcc_classifier.onnx` | 27.3 MB fp32 ONNX (6,764,880 params) |
| `config.json` | 프로토타입 뱅크·컷오프·출력 텐서 이름·스테이지 정의 |
| `examples/` | 예시 영상 8장 (HCC 4 + 혈관종 4) + 각 영상의 모델 입력 raw float32 + `expected.json` |
| `vendor/ort.wasm.min.js`, `vendor/ort-wasm-simd-threaded.wasm` | ONNX Runtime Web 1.23 (wasm, 11.3 MB) |
| `fonts/space_grotesk_*.ttf` | 앱과 같은 폰트 (OFL-1.1) |
| `qr-web-demo.png` | 위 URL 의 QR (752×752, error correction H) |

첫 방문 다운로드 ≈ 39 MB (모델 27 + wasm 11 + 예시 7 은 예시를 누를 때만). 이후 브라우저 캐시.

## 왜 ONNX Runtime Web 인가 (TensorFlow.js 가 아님)

모델의 exact GELU 는 `Erfc` 로 내려가는데 tfjs 에는 그 커널이 없음. 그래서

1. `erf` 형태(`0.5x(1+erf(x/√2))`)로 바꿔 SavedModel 을 저장 → 변환은 통과했지만
2. grappler 가 그 패턴을 다시 `Gelu(geluexact)` 로 융합 → tfjs 에 그 커널도 없음
3. 스케일을 변수로 바꿔 융합을 피하려 했지만 grappler 가 그 변수까지 상수로 접어버림

ONNX 는 `Erf` 가 표준 op 라 이 문제가 아예 없음. 다만 ONNX 도 `Erfc` 를 opset 15/17 에서
등록하지 않아서(`No Op registered for Erfc`) `erf` 형태로 저장함 — 아래 검증이 그 등가성을 수치로
확인함.

## 내보내기

```bash
python android_app/tools/export_web_onnx.py            # SavedModel → ONNX → 검증까지
python android_app/tools/export_web_onnx.py --skip-convert   # 페이지 작업만 반복할 때
```

스크립트는 변환 후 **onnxruntime 으로 4개 예시를 돌려** 앱이 쓰는 TFLite fp32 와 비교하고,
기준을 넘으면 exit 1 로 실패함:

```
[verify] PASS hcc1.jpg  dP=1.19e-07 dCos=2.27e-06 dMap=4.56e-06 dAttn=4.59e-06
[verify] PASS hcc2.jpg  dP=0.00e+00 dCos=1.49e-07 dMap=3.16e-06 dAttn=2.26e-06
[verify] PASS hcc5.jpg  dP=0.00e+00 dCos=7.30e-07 dMap=4.71e-06 dAttn=4.65e-06
[verify] PASS new_hcc  dP=0.00e+00 dCos=2.24e-07 dMap=1.85e-06 dAttn=2.68e-06
```

## 페이지의 두 가지 입력 경로

1. **예시 영상 8장 (HCC 4 + 혈관종 4)** — 페이지가 `examples/*.bin`(모델 입력 raw float32)을 그대로 먹임. 즉 앱과
   **완전히 같은 리샘플 경로**(bilinear + uint8, `dataloader.py`)를 씀. 아래 자체검증이 이 경로를 씀.
2. **방문자 사진** — 캔버스에서 384×384 로 리샘플만 하고 **보정 없이** 그대로 추론함. 웹 데모는
   모델 자체의 숫자를 보여주는 것이 목적이라, 보정(반사광 제거·denoise·조명 평탄화)은 앱에만 둠.
   화면에도 "보정 없음"으로 표시됨.

**주의**: 화살표·캘리퍼·주석·환자정보·장비 UI 가 조금이라도 들어가면 결과를 믿을 수 없음. 병변만
확대한 crop 도 안 됨(실측: 같은 혈관종 프레임을 0.65 crop 하면 HCC 3/3 로 뒤집힘). 페이지에 같은
경고를 카드로 띄워 둠.

**숫자 자체검증** 버튼을 누르면 방문자 기기에서 4개 예시를 다시 돌려 TFLite fp32 기준값과의 차이를
보여줌. 발표 중 "이 페이지가 자기 숫자를 검증한다"로 쓸 수 있음.

## 컷오프 — confidence 0.5 + 두 점수는 validation Youden (2026-10-23 변경)

| 지표 | 컷오프 | 근거 |
|---|---|---|
| `confidence_score` (P(HCC)) | **0.5** | 이진 분류의 관행 경계. 기존 0.0011 은 사실상 "항상 HCC" |
| `hcc_cosine_score` | **-0.004843** | 학습 코호트 validation Youden (유지) |
| `delta_score` (margin) | **-0.958471** | 학습 코호트 validation Youden (유지) |

SMC-LUD validation cohort 에서 딴 Youden 지점은 그 코호트에서 **AUC 1.0** 이 나왔고, 그 값을 외부
데이터에 그대로 쓰면 판정이 한쪽으로 붙는다(과적합 의심). 그래서 **확률은 관행 0.5 로 자르고, 순위
기반 점수 두 개는 기존 컷오프를 유지**한다 — 두 점수는 확률이 아니라 거리라 관행값이 없기 때문.
**어느 쪽도 외부 검증된 값은 아니며, 외부 코호트에서 재산출해야 한다.**

### 이 조합의 분류 성능 (실측)

`cosine_probe_result/raw_cosprobe_effnet_supcon.csv`, 주력 모델(EfficientNetV2B0 + CE + SupCon),
컷오프 고정: confidence ≥ 0.5, HCC cosine ≥ -0.004843, Δscore ≥ -0.958471.

```
Test n=268 (HCC 140 / 혈관종 128)
  Confidence P(HCC)  TP134 FP0 FN6 TN128  sens 95.71  spec 100.00  acc 97.76  PPV 100.00  NPV 95.52
  HCC cosine        TP140 FP1 FN0 TN127  sens 100.00 spec  99.22  acc 99.63  PPV  99.29  NPV 100.00
  Δscore (margin)   TP140 FP1 FN0 TN127  sens 100.00 spec  99.22  acc 99.63  PPV  99.29  NPV 100.00

Val n=530 (HCC 277 / 혈관종 253)
  Confidence P(HCC)  TP263 FP0 FN14 TN253 sens 94.95  spec 100.00  acc 97.36
  HCC cosine        TP277 FP0 FN0  TN253 sens 100.00 spec 100.00  acc 100.00
  Δscore (margin)   TP277 FP0 FN0  TN253 sens 100.00 spec 100.00  acc 100.00
```

- 세 출력이 모두 일치: Test 261/268 (97.4 %), Val 516/530 (97.4 %). 불일치는 **전부** confidence 만
  혈관종, 두 점수는 HCC 라고 하는 형태(Test 7건 = HCC 6 + 혈관종 1).
- 그래서 판정 규칙이 곧 operating point 다. **셋 다 HCC(AND)** → sens 95.7 / spec 100,
  **하나라도 HCC(OR)** → sens 100 / spec 99.2. 페이지의 "3/3 agree" 표시가 이 구분을 보여준다.
- 두 점수의 validation 성능 100 % 는 **컷오프를 그 세트에서 뽑았기 때문**(in-sample)이라 성능 추정치가
  아니다. Test 100/99.2 도 같은 코호트 안이라 낙관적이다.
- **AUC 1.0 은 순위가 완벽하다는 뜻**이다. 전역 보정은 나쁘지 않음(Brier 0.017 · ECE10 0.025, Test) —
  확률은 0 과 1 에 봉우리가 있는 이봉분포다. 문제는 **중간대(0.001–0.99)의 구성**이다: Test 24장 중 23장,
  Val 51장 중 51장이 HCC. 즉 경계 하위군에서 확률이 HCC 를 0 쪽으로 끌어내리는 반면, 같은 케이스의
  닮음 점수는 HCC 라고 답한다. 0.5 는 그래서 HCC 6장(0.028–0.404)을 놓친다.
  컷오프 재산출과 함께 이 중간대를 따로 보정(temperature scaling·isotonic)하는 편이 낫다.

외부 데이터가 생기면:

```bash
python3 recompute_cutoff.py external.json --labels external_labels.json          # 값만 확인
python3 recompute_cutoff.py external.json --labels external_labels.json --apply  # config.json 갱신
```

`--labels` 는 `{"파일명": "HCC"|"Hemangioma"}`, 생략하면 파일명에 `hcc`/`hem` 이 들어 있는지로 읽는다.
출력은 지표별 AUC · Youden J · 최적 컷오프 · 동률 구간이고, AUC 1.0 이면 "컷오프가 유일하지 않다"
경고를 낸다. **앱(`model_meta_*.json`)과 `inference/explainable_predictor.py` 는 아직 이전 값**이므로,
컷오프를 확정하면 세 곳을 같이 갱신해야 한다.

## 배포 (한 번만)

GitHub Pages 는 저장소 Settings → Pages → Source: **Deploy from a branch** → `main` / `/docs` 로
켜면 됨. 이 저장소는 public 이라 무료이고, 이미 public 인 h5 가중치와 비교해 새로 공개되는 것은
ONNX 사본뿐임.

## 발표 당일 체크리스트

1. 발표 전, 발표장 네트워크에서 **아이폰 1대 + 안드로이드 1대**로 QR 스캔 → 로딩 → 추론까지 리허설.
2. 슬라이드에 QR 과 **URL 텍스트를 같이** (QR 인식 실패 대비). QR 은 슬라이드 높이의 20 % 이상.
3. QR 을 쉬는 시간에 미리 띄워 39 MB 를 미리 받게 함. 동시 접속이 몰리면 회선이 병목이지 서버는 아님.
4. 예시 영상 슬라이드를 크게 띄워 두면 참가자가 **자기 폰으로 그 화면을 찍어** 넣는 흐름이 자연스러움.
5. 데모 하단 면책 문구(연구·교육용, 진단 아님)는 그대로 두고, 첫 화면에도 한 줄 안내.

## 검증 상태

- ONNX 변환·수치: `export_web_onnx.py` 가 매번 검증 (위 수치).
- 페이지: 헤드리스 Chromium 으로 로드 → 예시 추론 → 자체검증 버튼까지 실행해 확인함.
- 실기기(iOS Safari · Android Chrome)에서의 체감 속도와 첫 로딩 시간은 **검증 못 함** — wasm
  단일 스레드(COOP/COEP 없음)라 데스크톱보다 느림. 발표 전 리허설에서 실측할 것.
