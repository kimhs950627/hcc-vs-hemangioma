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

## 🚨 Critical Rules Before Training

### Rule 1 — `ssl_mode`와 Dataset은 반드시 짝을 맞춰야 한다

이 repo에서 가장 흔한 실수다. 아래 표를 반드시 확인하라.

| `ssl_mode` | **반드시 사용할 Dataset** | 잘못된 조합 |
|---|---|---|
| `'moco'` | `MultiViewDataset` | `MaskedMultiViewDataset` ❌ |
| `'byol'` | `MultiViewDataset` | `MaskedMultiViewDataset` ❌ |
| `'dino'` | `MultiViewDataset` | `MaskedMultiViewDataset` ❌ |
| `'dino_simmim'` | **`MaskedMultiViewDataset`** | `MultiViewDataset` ❌ |

잘못 짝지으면 첫 step에서 **view unpacking mismatch → shape 에러**로 crash한다.

### Rule 2 — Encoder는 반드시 `(224, 224, 3)` input으로 먼저 build해야 한다

`TrainablePositionalEmbedding`은 **처음 call되는 input shape**을 기준으로 `pos` weight를 build한다.  
만약 local crop (e.g. 112×112)이 teacher encoder에 먼저 들어오면 `pos_embed`가 49-patch 기준으로 build되고,  
이후 224×224 global view (196 patches)가 들어올 때 `_interpolate_pos`에서 reshape mismatch로 crash한다.

```python
# DINOSimMIMModel.__init__ 마지막 또는 trainer build 직후 반드시 실행
dummy = tf.zeros([1, 224, 224, 3])
_ = self.online_encoder(dummy, training=False)
_ = self.teacher_encoder(dummy, training=False)
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
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 4,       # n_local과 반드시 일치
    mask_ratio  = 0.75,    # SimMIM masking ratio
).as_dataset()
```

반환 view 순서 (train_step이 기대하는 순서):

| 인덱스 | 텐서 | 역할 |
|--------|------|------|
| `views[0]` | `original_clean` | Teacher global1 / SimMIM reconstruction target |
| `views[1]` | `aug_global2` | Teacher global2 / Student global2 (DINO) |
| `views[2]` | `masked_clean` | SimMIM student input |
| `views[3]` | `patch_mask` | `[B, N_patches]`, 1=masked 0=visible |
| `views[4:]` | `local_i × n` | DINO student local crops |

### 3. `MultiViewDataset` — legacy (dino / byol / moco 전용)

```python
from dataloader import MultiViewDataset

# DINO (global + local)
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
  view[0] original_clean ──► Teacher Encoder ──► DINO center  │
  view[1] aug_global2    ──┐                                   │
  view[4+] local_i      ──┴► Student Encoder ──► DINO loss    │
                                                               │
  view[2] masked_clean  ──► Student Encoder ──► Pixel Head    │
  view[3] patch_mask    ──────────────────────► SimMIM loss    │
          │                                                    │
          │   L_total = alpha * L_dino + (1-alpha)*λ*L_simmim │
          └────────────────────────────────────────────────────┘
```

- **DINO branch**: `original_clean`을 teacher global1로, `aug_global2`를 teacher/student global2로 사용.  
  local crops(`views[4:]`)는 student 전용이며 `online_encoder` 호출 경로를 공유한다.
- **SimMIM branch**: `masked_clean`을 student encoder에 통과시켜 patch token 추출 →  
  `pixel_pred_head`로 원본(`original_clean`) pixel 복원. `patch_mask` 위치만 L1 loss에 반영.
- **Alpha warm-up**: 초기 `alpha=1.0` (DINO 전용) → `warmup_steps` 동안 `alpha_final`까지 선형 감소.  
  `AlphaWarmupCallback`의 `on_train_batch_end`에서 step 단위로 갱신됨.

### Total Loss

$$L_{total} = \alpha \cdot L_{dino} + (1 - \alpha) \cdot \lambda_{mim} \cdot L_{simmim}$$

- `alpha`: warm-up 중 `1.0 → alpha_final` 선형 감소
- `lambda_mim`: SimMIM reconstruction loss weight (기본 `1.0`)

### Parameters

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `encoder_name` | str | — | `"vit"` / `"swin"` / `"convnext"` / `"efficientnet"` |
| `input_shape` | tuple | `(224,224,3)` | 이미지 입력 크기 |
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

# ── Step 1: Dataset ──────────────────────────────────────────────────────
mv_ds = MaskedMultiViewDataset(
    data_root   = "./clean_ver_for_train",
    split       = "train",
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 4,      # ← must equal n_local below
    mask_ratio  = 0.75,
).as_dataset()

mv_val_ds = MaskedMultiViewDataset(
    data_root   = "./clean_ver_for_train",
    split       = "val",
    img_size    = (224, 224),
    batch_size  = 16,
    local_views = 4,
    mask_ratio  = 0.75,
).as_dataset()

# ── Step 2: Build model ──────────────────────────────────────────────────
ssl_model = build_stage1_trainer(
    encoder_name         = "vit",
    input_shape          = (224, 224, 3),
    projection_dim       = 256,
    patch_size           = 16,
    n_local              = 4,         # ← must equal local_views above
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
    ssl_mode             = "dino_simmim",   # ← 핵심 인자
)

# ── Step 3: Warm-up pos_embed build (필수!) ──────────────────────────────
# 반드시 224×224 dummy로 먼저 build해야 pos_embed shape이 올바르게 설정됨
import tensorflow as tf
dummy = tf.zeros([1, 224, 224, 3])
_ = ssl_model.online_encoder(dummy, training=False)
_ = ssl_model.teacher_encoder(dummy, training=False)

# ── Step 4: Callbacks ────────────────────────────────────────────────────
steps_per_epoch = 116   # = ceil(train_size / batch_size)

callbacks = [
    AlphaWarmupCallback(
        warmup_steps = 20 * steps_per_epoch,   # 20 epochs
        verbose      = True,
    ),
    TeacherTempWarmupCallback(
        start_value  = 0.04,
        end_value    = 0.07,
        warmup_steps = 10 * steps_per_epoch,   # 10 epochs
        verbose      = True,
    ),
]

# ── Step 5: Train ────────────────────────────────────────────────────────
ssl_model.fit(
    mv_ds,
    validation_data = mv_val_ds,
    epochs          = 100,
    callbacks       = callbacks,
)

# ── Step 6: Export teacher encoder for Stage 2 ──────────────────────────
teacher_enc = ssl_model.get_stage2_encoder(use_teacher=True)
teacher_enc.save_weights("output/vit_stage1_dino_simmim.weights.h5")
```

### Loss 모니터링

학습 중 다음 metric이 기록된다.

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

### `TrainablePositionalEmbedding` — reshape mismatch crash

**증상**:
```
ValueError: Cannot reshape a tensor with 221184 elements to shape [1,14,14,384]
Arguments received by TrainablePositionalEmbedding.call():
  x=tf.Tensor(shape=(16, 577, 384), dtype=float32)
```

**원인**: `pos_embed`가 224×224 (196 patches) 기준으로 build됐는데,  
runtime에 384×384 (576 patches + CLS = 577) 입력이 teacher encoder에 들어감.

**해결**:
1. `DINOSimMIMModel.__init__` 이후 반드시 224×224 dummy warm-up 실행
2. `_interpolate_pos`에서 `h_t`, `w_t`를 `self.pos.shape[1]-1`의 integer sqrt로 재계산

```python
# models/encoder.py — _interpolate_pos 수정 예시
def _interpolate_pos(self, n_run: int, d: int) -> tf.Tensor:
    cls_pos   = self.pos[:, :1, :]
    patch_pos = self.pos[:, 1:, :]

    # stored weight의 실제 patch 수에서 h_t, w_t를 재계산
    n_stored = self.pos.shape[1] - 1      # static shape 사용
    h_t = int(n_stored ** 0.5)
    w_t = h_t

    n_patches_run = n_run - 1
    h_r = tf.cast(tf.math.round(tf.sqrt(tf.cast(n_patches_run, tf.float32))), tf.int32)
    w_r = tf.cast(tf.math.ceil(tf.cast(n_patches_run, tf.float32) / tf.cast(h_r, tf.float32)), tf.int32)

    patch_pos_2d = tf.reshape(patch_pos, [1, h_t, w_t, d])   # static shape → OK
    patch_pos_2d = tf.image.resize(patch_pos_2d, [h_r, w_r], method='bilinear')
    patch_pos_1d = tf.reshape(patch_pos_2d, [1, h_r * w_r, d])

    return tf.concat([cls_pos, patch_pos_1d], axis=1)
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

- **Transformer encoders** return: `embedding`, `last_hidden_state`, `last_encoder_layer_attentional_weights`
- **CNN encoders** return: `embedding`, `feature_map`
- `dino_simmim` SimMIM branch: `last_hidden_state[:, 1:, :]` (CLS 제거 후 patch tokens) 또는 `feature_map` reshape

---

## Augmentation Strategy

초음파 특화 photometric perturbation pipeline:

- `RandomContrast`, `RandomBrightness`, custom `RandomGamma`, `GaussianNoise`

Gain/TGC variability 및 hypoechoic lesion interior에 대한 shortcut learning을 억제한다.
