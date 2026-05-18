# HCC vs. Hemangioma Classification

B-mode 간 초음파 이미지에서 **간세포암종(HCC)** 과 **혈관종(Hemangioma)** 을 분류하는 딥러닝 프로젝트.

> **Dataset**: [SMC-LUD (Samsung Medical Center – Liver Ultrasound Dataset)](https://doi.org/10.6084/m9.figshare.31112716)  
> *Nature Scientific Data, 2026-03-10 · 5,385 B-mode images · 1,021 patients*

---

## ⭐ Recommended Default: DINO + SimMIM (`dino_simmim`)

> **이 repo에서 SSL pretraining의 기본 권장 mode는 `dino_simmim`이다.**

### Why `dino_simmim`?

| 특성 | DINO 단독 | SimMIM 단독 | **DINO + SimMIM (권장)** |
|---|---|---|---|
| Global representation | ✅ 강함 | ❌ 약함 | ✅ 강함 |
| Local structure | ❌ 약함 | ✅ 강함 | ✅ 강함 |
| Ultrasound 적합성 | 보통 | 보통 | **높음** |
| Required dataloader | `MultiViewDataset` | — | `MaskedMultiViewDataset` |

초음파 간 병변은 **전역 lesion semantics(DINO)** 와 **국소 텍스처 / 패치 구조(SimMIM)** 를 동시에 필요로 하기 때문에 joint training이 유리하다.

---

## 🔑 Resolution Contract — 가장 중요한 규칙

> **이 프로젝트의 모든 에러 중 가장 흔한 원인은 resolution(해상도) 불일치다.**

`MaskedMultiViewDataset`의 `img_size`, `build_stage1_trainer`의 `input_shape`, 각 view의 해상도 세 가지가 **항상 일치**해야 한다. 이 계약(contract)을 지키지 않으면 첫 step에서 crash한다.

### Resolution Contract 한눈에 보기

```
INPUT_SHAPE = (224, 224, 3)   ← 이 단 하나의 상수가 아래 모든 것을 결정한다
                 │
    ┌────────────┴────────────────────────────────────┐
    │                                                 │
MaskedMultiViewDataset                    build_stage1_trainer
    img_size = (224, 224)    ←──────────→  input_shape = (224, 224, 3)
    │
    ├── views[0]  original_clean   (224×224) → Teacher encoder  ✅
    ├── views[1]  masked_clean     (224×224) → Student encoder (SimMIM)  ✅
    ├── views[2]  patch_mask       [B, 196]  → SimMIM mask target  ✅
    ├── views[3]  aug_global       (224×224) → Student encoder (DINO)  ✅
    └── views[4+] local_i          (96×96)  → Student encoder only  ✅
                                             (teacher에는 절대 들어가지 않음)
```

### View별 해상도 규칙

| 인덱스 | 텐서 | 해상도 | 수신자 |
|--------|------|--------|--------|
| `views[0]` | `original_clean` | **= `img_size`** | Teacher encoder **only** |
| `views[1]` | `masked_clean`   | **= `img_size`** | Student encoder (SimMIM) |
| `views[2]` | `patch_mask`     | `[B, N_patches]` | SimMIM loss |
| `views[3]` | `aug_global`     | **= `img_size`** | Student encoder (DINO global) |
| `views[4+]`| `local_i`        | **< `img_size`** | Student encoder only |

> **Teacher encoder는 `original_clean`(= `img_size`) 하나만 받는다.**  
> Local crop은 student encoder에만 들어간다. 절대로 teacher에 전달하지 않는다.

### 코드에서 올바르게 적용하는 법

```python
INPUT_SHAPE = (224, 224, 3)   # ★ 이 상수 하나로 dataloader와 model을 동기화
LOCAL_VIEWS = 4               # ★ dataloader local_views = model n_local

mv_ds = MaskedMultiViewDataset(
    img_size    = INPUT_SHAPE[:2],   # (224, 224)
    local_views = LOCAL_VIEWS,
    ...
)

ssl_model = build_stage1_trainer(
    input_shape = INPUT_SHAPE,       # (224, 224, 3)
    n_local     = LOCAL_VIEWS,
    ...
)
```

---

## 🚨 Critical Rules Before Training

### Rule 1 — `ssl_mode`와 Dataset은 반드시 짝을 맞춰야 한다

| `ssl_mode` | **반드시 사용할 Dataset** | 잘못된 조합 |
|---|---|---|
| `'moco'` | `MultiViewDataset` | `MaskedMultiViewDataset` ❌ |
| `'byol'` | `MultiViewDataset` | `MaskedMultiViewDataset` ❌ |
| `'dino'` | `MultiViewDataset` | `MaskedMultiViewDataset` ❌ |
| `'dino_simmim'` | **`MaskedMultiViewDataset`** | `MultiViewDataset` ❌ |

### Rule 2 — Resolution: 모든 global view는 `input_shape`과 동일해야 한다

`views[0]`, `views[1]`, `views[3]`은 **전부 `img_size`** 와 동일한 해상도로 출력돼야 한다.  
`views[4+]` local crop만 더 작은 해상도를 가져도 된다.

```python
# ✅ 올바른 예
mv_ds = MaskedMultiViewDataset(img_size=(224, 224), ...)
ssl_model = build_stage1_trainer(input_shape=(224, 224, 3), ...)

# ❌ 잘못된 예 — resolution 불일치 → crash
mv_ds = MaskedMultiViewDataset(img_size=(384, 384), ...)
ssl_model = build_stage1_trainer(input_shape=(224, 224, 3), ...)
```

### Rule 3 — `n_local`은 `MaskedMultiViewDataset.local_views`와 반드시 일치해야 한다

```python
# ✅ 올바른 예
mv_ds = MaskedMultiViewDataset(..., local_views=4, ...)
ssl_model = build_stage1_trainer(..., n_local=4, ssl_mode='dino_simmim')

# ❌ 잘못된 예 → train_step view unpacking 오류
mv_ds = MaskedMultiViewDataset(..., local_views=2, ...)
ssl_model = build_stage1_trainer(..., n_local=4, ssl_mode='dino_simmim')
```

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Dataset Structure](#dataset-structure)
3. [Installation](#installation)
4. [Dataloader](#dataloader)
5. [Stage 1 — DINO + SimMIM SSL Trainer](#stage-1--dino--simmim-ssl-trainer)
6. [Stage 1 — Unified Router (`ssl_mode`)](#stage-1--unified-router-ssl_mode)
7. [Stage 1 — Other SSL Modes](#stage-1--other-ssl-modes)
8. [Stage 2 Supervised + SupCon](#stage-2-supervised--supcon-training)
9. [Supervised Benchmark Baseline](#supervised-benchmark-baseline)
10. [Prototype Bank Construction](#prototype-bank-construction)
11. [Inference and Scoring](#inference-and-scoring)
12. [W&B Visualization](#wb-visualization)
13. [Smoke-Test](#smoke-test)
14. [Known Issues & Debugging](#known-issues--debugging)

---

## Repository Structure

```
hcc-vs-hemangioma/
├── dataloader.py
├── models/
│   └── encoder.py           # VisionTransformer, Swin-like, ConvNeXt, EfficientNet
├── training/
│   ├── stage1_ssl.py         # Unified SSL router (moco/byol/dino/dino_simmim)
│   ├── stage1_dino.py        # DINOPretrainModel + DINOSimMIMModel
│   ├── stage1_moco.py
│   ├── stage1_byol.py
│   ├── stage2_supcon.py
│   └── benchmark_supervised.py
├── callbacks/
│   └── epoch_visualization.py
├── utils/
│   ├── metrics.py
│   └── gradcam.py
├── inference/
│   └── build_prototype_bank.py
├── visualization/
│   └── wandb_viz.py
└── README.md
```

---

## Dataset Structure

```
clean_ver_for_train/          ← data_root
├── train_clean/
│   ├── HCC/
│   └── Hemangioma/
├── val_clean/
│   ├── HCC/
│   └── Hemangioma/
└── test_clean/
    ├── HCC/
    └── Hemangioma/
```

| 클래스 | 레이블 |
|--------|--------|
| Hemangioma (혈관종) | `0` |
| HCC (간세포암종)    | `1` |

---

## Installation

```bash
pip install keras tensorflow numpy matplotlib
```

- **Keras 3** (backend-agnostic), Python 3.11+, single consumer GPU (≤16 GB VRAM)

---

## Dataloader

### Directory to Provide

```python
DATA_ROOT = "/path/to/clean_ver_for_train"
```

### 1. Supervised Dataloader (`build_dataset`)

```python
from dataloader import build_dataset

ds_train, ds_val, ds_test = build_dataset(
    data_root        = "/path/to/clean_ver_for_train",
    img_size         = (224, 224),
    batch_size       = 32,
    use_augmentation = True,
    seed             = 42,
)
```

### 2. `MaskedMultiViewDataset` — `dino_simmim` 전용 (권장)

`dino_simmim` mode에서는 반드시 이 dataloader를 사용해야 한다.

```python
from dataloader import MaskedMultiViewDataset

mv_ds = MaskedMultiViewDataset(
    data_root   = "/path/to/clean_ver_for_train",
    split       = "train",
    img_size    = (224, 224),   # ← 반드시 build_stage1_trainer input_shape과 동일
    batch_size  = 16,
    local_views = 4,            # ← n_local과 반드시 일치
    mask_ratio  = 0.75,
).as_dataset()
```

반환 view 순서 (train_step이 기대하는 순서):

| 인덱스 | 텐서 | 해상도 | 역할 |
|--------|------|--------|------|
| `views[0]` | `original_clean` | = `img_size` | Teacher global / SimMIM target |
| `views[1]` | `masked_clean` | = `img_size` | SimMIM student input |
| `views[2]` | `patch_mask` | `[B, N_patches]` | 1=masked, 0=visible |
| `views[3]` | `aug_global` | = `img_size` | DINO student global |
| `views[4:]` | `local_i × n` | < `img_size` | DINO student local crops (student only) |

> `views[0]`, `views[1]`, `views[3]`은 모두 `img_size`와 **동일한 해상도**여야 한다.  
> `views[4:]` local crops만 더 작은 해상도로 잘린다. **Local crop은 student encoder에만 들어간다.**

### 3. `MultiViewDataset` — legacy (dino / byol / moco 전용)

```python
from dataloader import MultiViewDataset

mv_ds = MultiViewDataset(
    data_root        = "/path/to/clean_ver_for_train",
    split            = "train",
    img_size         = (224, 224),
    batch_size       = 16,
    local_views      = 6,
    local_crop_scale = (0.05, 0.40),
)
```

### Augmentation Pipelines

| 함수 | 용도 | 강도 |
|------|------|------|
| `build_base_augmentation(img_size)` | Supervised train | 중간 |
| `build_strong_augmentation(img_size)` | SSL global view | 강 |
| `build_local_crop_augmentation(...)` | SSL local crop | 강+crop |

---

## Stage 1 — DINO + SimMIM SSL Trainer

`DINOSimMIMModel`은 **DINO self-distillation**과 **SimMIM masked image reconstruction**을  
단일 `train_step` 안에서 **동시에(jointly)** 최적화한다.

### 동작 원리

```
          ┌────────────────────────────────────────────────────┐
          │                  train_step                        │
          │                                                    │
  view[0] original_clean ──► Teacher Encoder ──► DINO target  │
  view[3] aug_global     ──┐                                   │
  view[4+] local_i      ──┴► Student Encoder ──► DINO loss    │
                                                               │
  view[1] masked_clean  ──► Student Encoder ──► Pixel Head    │
  view[2] patch_mask    ──────────────────────► SimMIM loss    │
          │                                                    │
          │   L_total = alpha*L_dino + (1-alpha)*λ*L_simmim   │
          └────────────────────────────────────────────────────┘
```

- **Teacher**: `original_clean` (= `img_size`) **만** 받는다. local crop은 절대 teacher에 들어가지 않는다.
- **Student**: `masked_clean`, `aug_global`, `local_i` 모두 받는다.
  - `masked_clean` / `aug_global`은 `img_size`, local crop은 더 작은 해상도.
- **Alpha warm-up**: 초기 `alpha=1.0` (DINO 전용) → `warmup_steps` 동안 `alpha_final`까지 선형 감소.

### Total Loss

$$L_{total} = \alpha \cdot L_{dino} + (1 - \alpha) \cdot \lambda_{mim} \cdot L_{simmim}$$

- `alpha`: warm-up 중 `1.0 → alpha_final` 선형 감소
- `lambda_mim`: SimMIM reconstruction loss weight (기본 `1.0`)

### Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `encoder_name` | str | — | `"vit"` / `"swin"` / `"convnext"` / `"efficientnet"` |
| `input_shape` | tuple | `(224,224,3)` | 이미지 입력 크기 — dataloader `img_size`와 반드시 일치 |
| `projection_dim` | int | `256` | DINO projection head output dim |
| `patch_size` | int | `16` | SimMIM patch 크기 (encoder와 일치) |
| `n_local` | int | `4` | local crop 수 (`MaskedMultiViewDataset.local_views`와 일치) |
| `lambda_mim` | float | `1.0` | SimMIM reconstruction loss weight |
| `alpha_final` | float | `0.7` | warm-up 완료 후 DINO 가중치 |
| `alpha_warmup_epochs` | int | `20` | router 전달용 목표 warm-up epoch |
| `temperature` | float | `0.1` | student softmax temperature |
| `teacher_temp` | float | `0.04` | teacher softmax temperature (초기값) |
| `center_momentum` | float | `0.9` | teacher center EMA momentum |
| `ema_momentum` | float | `0.996` | teacher weight EMA momentum |
| `lr` | float | `1e-4` | AdamW learning rate |
| `clipnorm` | float\|None | `1.0` | gradient clip norm |
| `weight_decay` | float | `1e-4` | AdamW weight decay |

---

## Stage 1 — Unified Router (`ssl_mode`)

`build_stage1_trainer()`는 `ssl_mode` 인자 하나로 모든 SSL 방식을 라우팅하는 **통합 진입점**이다.

### 🔑 `ssl_mode='dino_simmim'` — Full DINO + SimMIM Training (권장)

```python
from dataloader import MaskedMultiViewDataset
from training.stage1_ssl import build_stage1_trainer
from training.stage1_dino import AlphaWarmupCallback
from training.ssl_callbacks import TeacherTempWarmupCallback

INPUT_SHAPE = (224, 224, 3)   # ★ 이 값을 dataloader와 model 양쪽에 동일하게 쓴다
LOCAL_VIEWS = 4               # ★ 이 값을 dataloader와 model 양쪽에 동일하게 쓴다

# ── Step 1: Dataset ──────────────────────────────────────────────────────
mv_ds = MaskedMultiViewDataset(
    data_root   = "./clean_ver_for_train",
    split       = "train",
    img_size    = INPUT_SHAPE[:2],   # (224, 224) — model input_shape과 반드시 동일
    batch_size  = 16,
    local_views = LOCAL_VIEWS,
    mask_ratio  = 0.75,
).as_dataset()

mv_val_ds = MaskedMultiViewDataset(
    data_root   = "./clean_ver_for_train",
    split       = "val",
    img_size    = INPUT_SHAPE[:2],
    batch_size  = 16,
    local_views = LOCAL_VIEWS,
    mask_ratio  = 0.75,
).as_dataset()

# ── Step 2: Build model ──────────────────────────────────────────────────
ssl_model = build_stage1_trainer(
    encoder_name         = "vit",
    input_shape          = INPUT_SHAPE,   # (224, 224, 3) — dataloader img_size와 동일
    projection_dim       = 256,
    patch_size           = 16,
    n_local              = LOCAL_VIEWS,
    lambda_mim           = 1.0,
    alpha_final          = 0.7,
    alpha_warmup_epochs  = 20,
    temperature          = 0.1,
    teacher_temp         = 0.04,
    center_momentum      = 0.9,
    ema_momentum         = 0.996,
    lr                   = 1e-4,
    clipnorm             = 1.0,
    weight_decay         = 1e-4,
    ssl_mode             = "dino_simmim",
)

# ── Step 3: Callbacks ────────────────────────────────────────────────────
steps_per_epoch = 116   # = ceil(train_size / batch_size)

callbacks = [
    AlphaWarmupCallback(
        warmup_steps = 20 * steps_per_epoch,
        verbose      = True,
    ),
    TeacherTempWarmupCallback(
        start_value  = 0.04,
        end_value    = 0.07,
        warmup_steps = 10 * steps_per_epoch,
        verbose      = True,
    ),
]

# ── Step 4: Train ────────────────────────────────────────────────────────
ssl_model.fit(
    mv_ds,
    validation_data = mv_val_ds,
    epochs          = 100,
    callbacks       = callbacks,
)

# ── Step 5: Export teacher encoder for Stage 2 ──────────────────────────
teacher_enc = ssl_model.get_stage2_encoder(use_teacher=True)
teacher_enc.save_weights("output/vit_stage1_dino_simmim.weights.h5")
```

### Loss 모니터링

```
loss          — total: alpha*L_dino + (1-alpha)*lambda_mim*L_simmim
l_dino        — DINO cross-entropy distillation loss
l_simmim      — SimMIM masked-patch L1 reconstruction loss
alpha         — current alpha blend ratio (step-wise: 1.0 → alpha_final)
teacher_temp  — current teacher softmax temperature
```

### SSL mode 비교

| `ssl_mode` | 필요 Dataset | Loss | 추천 encoder |
|---|---|---|---|
| `'moco'` | `MultiViewDataset` (2 global) | InfoNCE | vit / convnext |
| `'byol'` | `MultiViewDataset` (2 global) | MSE bootstrapping | vit / convnext |
| `'dino'` | `MultiViewDataset` (2 global + N local) | DINO cross-entropy | vit |
| `'dino_simmim'` | `MaskedMultiViewDataset` (2 global + masked + N local) | DINO + SimMIM L1 | **vit 전용** |

> CNN (`convnext`, `efficientnet`)은 `last_hidden_state` key를 반환하지 않아  
> SimMIM branch에서 `feature_map` reshape 경로를 탄다. **ViT 사용을 강력히 권장한다.**

---

## Stage 1 — Other SSL Modes

```python
from training.stage1_ssl import build_stage1_trainer

# MoCo
moco_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, temperature=0.1, ema_momentum=0.996,
    lr=1e-4, ssl_mode="moco",
)

# BYOL
byol_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, predictor_dim=256, ema_momentum=0.996,
    lr=1e-4, ssl_mode="byol",
)

# DINO only
dino_model = build_stage1_trainer(
    encoder_name="vit", input_shape=(224,224,3),
    projection_dim=256, temperature=0.1,
    teacher_temp=0.04, center_momentum=0.9, ema_momentum=0.996,
    n_local=4, lr=1e-4, ssl_mode="dino",
)
```

---

## Stage 2 Supervised + SupCon Training

```python
from dataloader import build_dataset
from training.stage2_supcon import build_stage2_trainer

ds_train, ds_val, ds_test = build_dataset(
    data_root="./clean_ver_for_train",
    img_size=(224,224), batch_size=16, use_augmentation=True,
)

stage2 = build_stage2_trainer(
    encoder_name              = "vit",
    input_shape               = (224, 224, 3),
    num_classes               = 2,
    supcon_weight             = 0.3,
    projection_dim            = 128,
    classifier_hidden_dim     = 256,
    dropout_rate              = 0.2,
    lr                        = 1e-4,
    teacher_encoder_weights   = "output/vit_stage1_dino_simmim.weights.h5",
)

stage2.fit(ds_train, validation_data=ds_val, epochs=50)
stage2.model.save_weights("output/vit_stage2.weights.h5")
```

---

## Supervised Benchmark Baseline

**No pretraining. No regularizer. No dropout. Scratch supervised binary classification.**  
Goal: establish a clean lower-bound to compare against SSL-pretrained models.

### CLI

```bash
# ViT scratch
python training/benchmark_supervised.py \
    --data_root ./clean_ver_for_train \
    --encoder vit --epochs 100 --batch_size 32 --lr 1e-4 \
    --output_dir outputs/benchmark_vit_scratch

# ConvNeXt with ImageNet pretrained encoder
python training/benchmark_supervised.py \
    --data_root ./clean_ver_for_train \
    --encoder convnext --epochs 100 --batch_size 16 --lr 1e-4 \
    --use_pretrained \
    --output_dir outputs/benchmark_convnext_pretrained
```

### Python API

```python
from training.benchmark_supervised import run_supervised_benchmark

result = run_supervised_benchmark(
    data_root      = "./clean_ver_for_train",
    encoder_name   = "vit",
    input_shape    = (224, 224, 3),
    batch_size     = 32,
    epochs         = 100,
    lr             = 1e-4,
    output_dir     = "outputs/benchmark_vit",
)
print(result)
```

### Metrics

| Metric | Formula |
|---|---|
| Accuracy | (TP + TN) / N |
| ROC-AUC | Trapezoidal rule |
| Sensitivity | TP / (TP + FN) |
| Specificity | TN / (TN + FP) |
| PPV | TP / (TP + FP) |
| NPV | TN / (TN + FN) |
| F1 | 2 × PPV × Sensitivity / (PPV + Sensitivity) |
| Youden threshold | argmax(sensitivity + specificity − 1) |

---

## Prototype Bank Construction

```bash
python inference/build_prototype_bank.py \
  --data_root ./clean_ver_for_train \
  --encoder   vit \
  --weights   output/stage2_classifier.weights.h5 \
  --n_prototypes 8 \
  --output_dir   output/prototype_bank
```

---

## Inference and Scoring

```python
from models.encoder import build_classifier

model = build_classifier("vit", input_shape=(224, 224, 3))
model.load_weights("output/stage2_classifier.weights.h5")
```

---

## W&B Visualization

```python
import wandb
from visualization.wandb_viz import init_wandb, get_wandb_callbacks

init_wandb(project="hcc-vs-hemangioma", run_name="stage1-dino-simmim")
```

---

## Smoke-Test

```bash
python dataloader.py /path/to/clean_ver_for_train
```

정상 실행 시 Train/Val/Test split 확인 + shape/dtype/range 검증 출력.

---

## Known Issues & Debugging

### Resolution mismatch crash — 가장 흔한 에러

**증상**:
```
ValueError: Exception encountered when calling PositionalEmbedding.call().
Cannot reshape a tensor with 221184 elements to shape [1,14,14,384]
  x=tf.Tensor(shape=(16, 577, 384), dtype=float32)
```

**에러 해석**:

| 관찰 값 | 의미 |
|---|---|
| `shape=(16, 577, 384)` | 577 = 576 patches + 1 CLS → **입력이 384×384** (24×24 grid) |
| `[1,14,14,384]` reshape 시도 | 14×14 = 196 patches → **pos_embed는 224×224로 build됨** |
| 결론 | dataloader `img_size=(384,384)` vs model `input_shape=(224,224,3)` 불일치 |

**해결 체크리스트**:

```python
# ✅ 1. INPUT_SHAPE 상수 하나로 통일
INPUT_SHAPE = (224, 224, 3)

mv_ds = MaskedMultiViewDataset(
    img_size    = INPUT_SHAPE[:2],   # (224, 224)
    ...
)
ssl_model = build_stage1_trainer(
    input_shape = INPUT_SHAPE,       # (224, 224, 3)
    ...
)

# ✅ 2. Teacher encoder 수신 view 확인
# DINOSimMIMModel.train_step 에서:
#   teacher → original_clean (= img_size, 224×224) 만 받음
#   local crop (96×96) 은 student encoder 에만 들어감
```

### `n_local` mismatch → `train_step` view unpacking 오류

**증상**: IndexError 또는 silent wrong-view feeding.  
**해결**: `MaskedMultiViewDataset(local_views=N)` = `build_stage1_trainer(n_local=N)` 항상 동일하게.

### patch_mask shape mismatch

**증상**: `_simmim_forward()` 내 `tf.debugging.assert_equal` 실패.  
**확인**: `patch_size=16`, `input_shape=(224,224,3)` → `N = (224/16)^2 = 196` patches.  
`patch_mask.shape[1]` == 196 이어야 한다.

---

## Notes on Outputs

- **Transformer encoders** return: `embedding`, `encoded_patches`, `last_encoder_layer_attentional_weights`
- **CNN encoders** return: `embedding`, `feature_map`
- `dino_simmim` SimMIM branch: `encoded_patches` (CLS 제거된 patch tokens, shape `(B, N, D)`) → `mim_head` 통과

---

## Augmentation Strategy

초음파 특화 photometric perturbation pipeline:

- `RandomContrast`, `RandomBrightness`, custom `RandomGamma`, `GaussianNoise`

Gain/TGC variability 및 hypoechoic lesion interior에 대한 shortcut learning을 억제한다.


## Stage1 SSL usage

This repository supports `DINO`, `DINO + SelfPatch`, `DINO + SimMIM`, and `DINO + SimMIM + SelfPatch` in stage1 pretraining.

### 1) DINO only
```python
from training.stage1_dino import build_stage1_dino_trainer

model = build_stage1_dino_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    n_local=6,
    encoder_embed_dim=384,
    encoder_depth=8,
    encoder_num_heads=8,
    encoder_mlp_dim=1536,
    encoder_patch_size=16,
    use_pe=True,
    lambda_selfpatch=0.0,
    lr=1e-4,
)
```

### 2) DINO + SelfPatch
```python
from training.stage1_dino import build_stage1_dino_trainer

model = build_stage1_dino_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    n_local=6,
    encoder_embed_dim=384,
    encoder_depth=8,
    encoder_num_heads=8,
    encoder_mlp_dim=1536,
    encoder_patch_size=16,
    use_pe=True,
    lambda_selfpatch=0.05,
    selfpatch_proj_dim=256,
    selfpatch_top_k=4,
    selfpatch_temperature=0.07,
    lr=1e-4,
)
```

### 3) DINO + SimMIM
```python
from training.stage1_dino_simmim import build_stage1_dino_simmim_trainer

model = build_stage1_dino_simmim_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    n_local=6,
    lambda_simmim=0.30,
    patch_size=16,
    encoder_embed_dim=384,
    encoder_depth=8,
    encoder_num_heads=8,
    encoder_mlp_dim=1536,
    encoder_patch_size=16,
    use_pe=True,
    lambda_selfpatch=0.0,
    lr=1e-4,
)
```

### 4) DINO + SimMIM + SelfPatch
```python
from training.stage1_dino_simmim import build_stage1_dino_simmim_trainer

model = build_stage1_dino_simmim_trainer(
    encoder_name="vit",
    input_shape=(224, 224, 3),
    projection_dim=256,
    n_local=6,
    lambda_simmim=0.30,
    patch_size=16,
    encoder_embed_dim=384,
    encoder_depth=8,
    encoder_num_heads=8,
    encoder_mlp_dim=1536,
    encoder_patch_size=16,
    use_pe=True,
    lambda_selfpatch=0.05,
    selfpatch_proj_dim=256,
    selfpatch_top_k=4,
    selfpatch_temperature=0.07,
    lr=1e-4,
)
```

### Notes
- `lambda_selfpatch == 0.0` means SelfPatch is disabled.
- `lambda_selfpatch > 0.0` means SelfPatch is enabled as an auxiliary patch loss.
- `lambda_simmim > 0.0` enables SimMIM in the hybrid trainer.
- For the hybrid trainer, the dataset should yield `(original_clean, masked_clean, patch_mask, aug_global, local_1, ..., local_N)`.
