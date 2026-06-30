# hcc-vs-hemangioma

> **B-mode Ultrasound 기반 HCC Score 개발 — Operator-Independent Radiologic Marker**  
> Primary Care AI · Computer Vision · Self-Supervised Learning · Radiologic Marker Development

---

## 🔬 Research Motivation

일차의료(primary care) 현장에서 PIVKA-II, AFP 등 종양표지자를 폭넓게 검사하기 어려운 현실이 있다.  
비용, 보험 급여, 추적 동선 등의 문제로 인해 **tumor marker 없이도 HCC를 스크리닝할 수 있는 도구**가 필요하다.

진료실에서 이미 사용 중인 **B-mode Ultrasound 이미지**만을 입력으로 받아,  
딥러닝 기반 **HCC Score (0–1 continuous)** 를 산출하는 Computer Vision 모델을 개발한다.  
이 score는 radiologic marker로 기능하며, 명확한 cutoff와 함께 임상적 의사결정을 지원한다.

---
# 실험 정리

| # | Mode                    | Backbone         | Model ID                                 | Status |
| - | ----------------------- | ---------------- | ---------------------------------------- | ------ |
| 1 | Classification Only     | ResNet50V2       | Benchmark-Classification_Only_resnet50v2 | ✅ 완료   |
| 2 | Classification Only     | EfficientNetV2B0 | Benchmark-Classification_Only            | ✅ 완료   |
| 3 | Classification + SupCon | ResNet50V2       | ke8bqdcv                                        | ✅ 완료 |
| 4 | Classification + SupCon | EfficientNetV2B0 | znkaz53c                                        | ✅ 완료 |
| 5 | VICReg (SSL)            | ResNet50V2       | —                                        | 🔲 미실행 |
| 6 | VICReg (SSL)            | EfficientNetV2B0 | 6bn40bha                                 | ✅ 완료   |
| 7 | NNCLR (SSL)             | ResNet50V2       | 7p28wnk2                                 | ✅ 완료   |
| 8 | NNCLR (SSL)             | EfficientNetV2B0 | zqe3125v                                 | ✅ 완료   |

## 🧠 Model Architecture Overview

### SSL Pre-training: VICReg + Hybrid ViT

본 연구는 **DINO를 사용하지 않는다.** 대신 아래 구조를 채택한다.

```
┌─────────────────────────────────────────────────────┐
│               Hybrid Vision Transformer              │
│                                                     │
│  Input 384×384 (grayscale, [0,255])                  │
│       ↓                                             │
│  CNN Stem (Conv 7×7, stride 4)                       │
│       ↓  Patch embedding via CNN (not linear proj)   │
│  Transformer Encoder (ViT-S/16 blocks)               │
│       ↓  [CLS] token representation                  │
│  Projector MLP (2048-2048-2048, BN, ReLU)            │
│       ↓                                             │
│  VICReg Loss: Variance + Invariance + Covariance     │
└─────────────────────────────────────────────────────┘
         ↓  (SSL pre-training finished)
┌─────────────────────────────────────────────────────┐
│          Supervised Fine-tuning (PureClassifier)     │
│                                                     │
│  Hybrid ViT Encoder (frozen or partial fine-tune)   │
│       ↓  [CLS] token                               │
│  Linear head → softmax(2) → [P(Hem), P(HCC)]        │
│       ↓                                             │
│  HCC Score = P(HCC) ∈ [0, 1]                        │
└─────────────────────────────────────────────────────┘
```

### Why Hybrid ViT (not plain ViT)?

| Feature | Plain ViT | Hybrid ViT (ours) |
|---|---|---|
| Patch embedding | Linear projection | CNN stem (Conv 7×7 + stride) |
| Low-level feature | Weak (needs huge data) | Strong (CNN inductive bias) |
| Small dataset fit | Poor | **Better** |
| Positional encoding | Fixed/learned 1D | CNN feature map → 2D spatial |
| Memory (384×384) | High (576 patches) | Moderate (CNN reduces spatial early) |

> **의료 이미징 맥락**: labeled 의료 데이터는 ImageNet 대비 절대적으로 적다.  
> CNN stem의 inductive bias (locality, translation equivariance)는  
> 초음파 병변의 texturally similar한 특징 추출에 효과적이다.

### Why VICReg (not DINO)?

| 항목 | DINO | VICReg |
|---|---|---|
| Architecture | Teacher-Student (EMA) | Symmetric twin network |
| Loss | Cross-entropy (softmax) | Variance + Invariance + Covariance |
| Collapse 방지 | Centering + Sharpening | Variance regularization term |
| Momentum update | ✅ 필요 | ❌ 불필요 (simpler) |
| Batch sensitivity | 높음 | **낮음** (small batch 가능) |
| Feature redundancy 제거 | 암묵적 | **명시적** (Covariance term) |
| 의료 US 적합성 | 검증 있으나 복잡 | **구현 단순, 소규모 데이터 유리** |

> VICReg의 핵심: 세 loss의 합산으로 collapse 방지 + decorrelated feature 학습  
> `L = λ·Invariance + μ·Variance + ν·Covariance`  
> (default: λ=25, μ=25, ν=1 — Bardes et al., NeurIPS 2022)

---

## 📊 Dataset

```
Modality     : B-mode Ultrasound (grayscale)
Task         : Binary classification — HCC vs Hemangioma
Total images : 2,656
  Train      : 1,858  (HCC 972 / Hemangioma 886)
  Val        :   530  (HCC 277 / Hemangioma 253)
  Test       :   268  (HCC 140 / Hemangioma 128)

Image spec   : 384 × 384 px, float32 [0, 255], normalize=False
Label        : 0 = Hemangioma, 1 = HCC (positive class)
Class ratio  : ~52% HCC / ~48% Hemangioma (balanced)
```

---

## ⚙️ Experimental Pipeline

### Stage 1 — VICReg SSL Pre-training

```python
# Pseudo-code (Keras 3 / TF backend)
class HybridViTEncoder(keras.Model):
    """
    CNN Stem: Conv2D(96, 7, stride=4) → LayerNorm
    Transformer Encoder: depth=12, heads=6, dim=384 (ViT-S config)
    Output: [CLS] token, shape (B, 384)
    """

class VICRegProjector(keras.Model):
    """
    MLP: 384 → 2048 → 2048 → 2048 (BN + ReLU, last layer no activation)
    Output: (B, 2048)
    """

# VICReg Loss
def vicreg_loss(z1, z2, lam=25.0, mu=25.0, nu=1.0):
    inv = mse_loss(z1, z2)                        # Invariance
    var = variance_loss(z1) + variance_loss(z2)   # Variance
    cov = covariance_loss(z1) + covariance_loss(z2)  # Covariance
    return lam * inv + mu * var + nu * cov
```

**Augmentation (SSL pre-training)**:
- Random crop + resize to 384×384
- Random horizontal/vertical flip
- Gaussian blur, brightness/contrast jitter (grayscale-safe)
- **Two views** per image → (z1, z2) → VICReg loss

### Stage 2 — Supervised Fine-tuning (PureClassifier)

```python
class PureClassifier(keras.Model):
    """
    encoder : HybridViTEncoder (pretrained, partial fine-tune or frozen)
    head    : Dense(2) → Softmax
    output  : dict {"logits": (B,2), "probabilities": (B,2)}
    """
    def call(self, x, training=False):
        feats = self.encoder(x, training=training)   # (B, 384)
        logits = self.head(feats)
        return {"logits": logits,
                "probabilities": tf.nn.softmax(logits)}
```

**HCC Score 정의**:

```
HCC Score = probabilities[:, 1] = P(HCC)  ∈ [0, 1]
```

### Stage 3 — Radiologic Marker Cutoff

| Strategy | Formula | Set |
|---|---|---|
| **Youden's J** (primary) | argmax(Sensitivity + Specificity − 1) | **Val only** |
| Sensitivity-first | min threshold s.t. Sensitivity ≥ 0.90 | Val only |

> **원칙**: Cutoff는 Val set에서만 결정. Test는 완전 blind evaluation.

### Stage 4 — Statistical Analysis

- AUROC: DeLong method 95% CI
- At cutoff: Sens, Spec, PPV, NPV, F1, Confusion Matrix
- **Decision Curve Analysis (DCA)**: Vickers & Elkin 2006
- Bootstrap (n=1,000): 95% CI for all metrics
- Score distribution: per-class Gaussian KDE (Train / Val / Test 3-row subplot)

---

## 🗂️ Repository Structure

```
hcc-vs-hemangioma/
├── README.md                           ← 실험·방법론 위주 요약 (this file)
├── paper_outline_medical_journal.md    ← 논문 뼈대 (Intro ~ Conclusion)
├── Stage1_SSK_run.ipynb                ← SSL pre-training (VICReg)
├── Stage2_Classification_Benchmark.ipynb ← Supervised fine-tuning benchmark
├── VICReg_ConvHybrid_run.ipynb         ← Hybrid ViT + VICReg 실험
├── VICReg_Conv_run.ipynb               ← ConvNet + VICReg 실험
├── full_training_notebook.ipynb        ← 통합 학습 노트북
├── dataloader.py                       ← Dataset pipeline
├── models/                             ← Model 정의
├── training/                           ← Training loop, loss
├── validation/                         ← Val/Test evaluation
├── inference/                          ← Score extraction, cutoff
├── visualization/                      ← ROC, KDE, DCA 시각화
├── callbacks/                          ← Keras callbacks
├── utils/                              ← 공통 유틸
└── experiment_registry/                ← 실험 결과 기록
```

---

## 🚀 Future Work

| Priority | Task | Description |
|---|---|---|
| **High** | AFP / PIVKA-II 연동 | Serology data와 B-mode US image를 paired하여 AFP-negative subgroup 성능 검증 |
| **High** | 다기관 전향적 연구 | External validation (multi-center) → generalizability 확인 |
| **Medium** | 영상의학과 vs HCC Score | Human radiologist performance와 직접 비교 (non-inferiority study) |
| **Medium** | <2cm subgroup 분석 | Lesion size metadata 확보 후 early-stage HCC 집중 분석 |
| **Low** | FNH / regenerative nodule 추가 | 대조군 확장 → 더 현실적인 임상 시나리오 |
| **Low** | CEUS 비교 | B-mode 단독 vs CEUS보조 성능 비교 |

---

## 🛠️ Tech Stack

| Component | Choice | Rationale |
|---|---|---|
| Framework | Keras 3 (TF backend) | Backend-agnostic, clean API |
| SSL Algorithm | **VICReg** | No momentum network, small-batch friendly, explicit decorrelation |
| Backbone | **Hybrid ViT** (CNN stem + Transformer) | CNN inductive bias + global attention |
| Fine-tuning | PureClassifier (linear head) | SSL representation quality 평가 목적 |
| Environment | Kaggle Notebooks (GPU T4 x2) | 16 GB VRAM constraint |
| Analysis | scikit-learn, scipy, matplotlib | ROC, DCA, KDE |
| Reporting | TRIPOD guideline | Prediction model reporting standard |

---

## 📝 Author

**Kim Hyun-Soo, M.D.**  
Department of Family Medicine, Primary Care Clinic, Jeonju, Republic of Korea  
AI Researcher (amateur) · Medical AI · Multimodal Learning · CV · Self-Supervised Learning  
GitHub: [kimhs950627](https://github.com/kimhs950627)

---

*Last updated: 2026-06-25*
