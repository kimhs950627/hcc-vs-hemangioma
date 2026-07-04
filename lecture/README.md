# HCC vs Hemangioma — 학회 발표 자료

의사 청중을 대상으로 한 B-mode 초음파 HCC 이중 출력 영상표지자 연구 발표 자료.

## 파일

- `hcc_hemangioma_lecture.html` — reveal.js 기반 발표 슬라이드 (29장)
- `hcc_hemangioma_lecture.pdf` — PDF 변환본 (29페이지)
- `hcc_hemangioma_lecture.pptx` — PowerPoint 변환본 (29슬라이드)
- `figures/` — 슬라이드에 삽입된 figure (t-SNE, confusion matrix, ROC, SupCon vs CE 산점도 등)

## 내용 개요

1. 연구 배경 (초음파 감시 한계, 클래스 내 이질성, 임상 추론, 기존 AI 한계)
2. 연구의 새로움 (Novelty) 및 연구 개요
3. 대상 및 방법 (데이터, CNN/ViT/임베딩 기초, 모델 구조, Cosine Score 산출, 손실 함수, 학습 전략)
4. 결과 (주력 모델 성능, 혼동행렬, t-SNE, Triple ROC, SupCon 필요성, salvage 효과, SSL 한계)
5. 고찰 (임상적 의의, 한계 및 향후 연구)
6. 결론 및 Q&A

## 사용 모델

EfficientNetV2B0 말단에 4겹의 Transformer encoder를 부착한 hybrid 구조.
학습: 교차 엔트로피(CE) + Supervised Contrastive Learning(SupCon).
출력: Confidence score / HCC Cosine score / Δscore.

## 렌더링

HTML은 reveal.js + MathJax CDN을 사용하므로 온라인에서 열어야 수식·레이아웃이 정상 렌더링됩니다.
오프라인 배포 시 PDF 또는 PPTX를 사용하세요.
