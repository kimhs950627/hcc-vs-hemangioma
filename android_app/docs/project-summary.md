# 기획서 — LiverDx (HCC vs Hemangioma US 판독 보조, 디버그 앱)

## 한 줄 정의
B-mode 간초음파 사진 한 장을 넣으면 **HCC vs Hemangioma 라벨 · Confidence Score ·
HCC/Hemangioma Cosine Score · Δscore · attention/프로토타입 히트맵**을 즉시 보여주는 Android 앱.

## 대상 사용자
- 1차 사용자: 연구자 본인 (가정의학과 전문의, 검진센터 위·대장내시경 + 복부초음파).
- 용도: **디버깅/시연**. 논문 리비전·발표 때 노트북 없이 폰으로 증례를 확인하고 CAM/attention을 보여주기.
- 2차 사용자: 공동연구자 (비임상 개발자). 점수 정의를 눈으로 확인할 수 있어야 함.

## 성공 기준 (acceptance)
1. `classifier_y0taaeki_BM.weights.h5` (EfficientNetV2B0 + CE + SupCon)의 추론 결과가 폰에서 재현된다.
   - Python `inference/explainable_predictor.py` 와 **같은 이미지에 대해 라벨이 일치**.
   - Confidence / HCC Cosine / Hemangioma Cosine / Δscore 가 3자리까지 일치 (허용 오차 ≤ 0.01).
2. 촬영(시스템 카메라) 또는 갤러리 선택한 사진이 384×384 로 전처리되어 추론된다.
3. Attention rollout 맵과 HCC 프로토타입 유사도 맵이 원본 위에 오버레이로 보인다.
3b. 앱 내 카메라로 촬영한 폰 사진에서 반사광·조명 얼룩·색조·노이즈를 제거한 뒤 추론하고, RAW ↔
   CLEANED 를 눈으로 비교할 수 있다. 각 단계는 개별 토글되고 토글 시 즉시 재분석된다.
3c. 리샘플 방식(bilinear+uint8 / area)을 선택할 수 있고 현재 방식이 화면에 표시된다.
4. `assembleDebug` 로 APK 가 만들어지고, 실기기에 adb install 이 된다.
5. 상단 모델 칩에 학습 구성·embedding 차원·patch grid·입력 해상도·가중치 sha256 앞 10자가
   표시되고, 각 점수행에 컷오프 값이 함께 표시된다 (재현성).

## 하지 않을 것 (negative constraints)
- **Grad-CAM 없음.** TFLite 는 gradient 를 지원하지 않는다. 대신 CA attention rollout +
  HCC prototype similarity 를 시각화한다. (Python 쪽 Grad-CAM 은 그대로 유지)
- **학습·미세조정 없음.** 추론 전용.
- **PACS/DICOM 연동, 환자정보 저장 없음.** 촬영본은 캐시의 단일 파일 하나로만 존재하고 디코드
  직후 삭제된다. 네트워크 전송 없음 (매니페스트에 INTERNET·CAMERA 권한 없음).
- **의료기기 아님.** 화면 하단 고정 면책 문구. 진단에 사용 금지.
- **다국어 없음.** 한국어 + 영문 병기.
- **Gradle release 서명/AAB 없음.** debug APK 와 debug 키로 서명한 경량 release APK 만.
- **fp16 을 기본으로 쓰지 않음.** Δcosine 9.5e-3 이 측정되어 기본은 fp32, 경량 빌드에서만 선택.
- **카메라 자동 보정(기하 왜곡·원근) 없음.** 화면을 비스듬히 찍은 사진의 원근은 보정하지 않는다.
  프레이밍 가이드로만 안내하고, 보정은 사용자 몫이다.

## 제약
- minSdk 26 (adaptive icon + Compose + variable font), compileSdk/targetSdk 35.
- CameraX 1.4.1 로 앱 내 촬영. CAMERA 런타임 권한 1개만 요청한다 (INTERNET 없음).
- 모델 자산은 fp32 TFLite 약 27 MB. dynamic-range quantization 은 128-d embedding 을
  최대 2.8e-1 흔들어 cosine score 를 바꾸므로 **사용하지 않는다** (측정 근거: `tools/export_tflite.py`).
- 앱 크기 목표 ≤ 60 MB (debug).
