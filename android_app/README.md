# LiverDx — HCC vs Hemangioma B-mode US (Android, debug)

On-device debug reader for the paper's hybrid ViT. Shoot a B-mode liver ultrasound frame with the
in-app camera (or pick one), the phone cleans the photo up, and you get the four paper markers plus
two spatial maps.

| Output | Source |
|---|---|
| Label (HCC / Hemangioma) | softmax argmax |
| Confidence Score | softmax P(HCC), compared with the validation cut-off |
| HCC Cosine Score | cosine of the 128-d CLS embedding to the mean HCC prototype |
| Hemangioma Cosine Score | cosine to the mean hemangioma prototype |
| Δscore | HCC cosine − hemangioma cosine |
| CA attention rollout | head-mean, layer-weighted rollout of the 4 cross-attention blocks |
| HCC prototype map | per-patch mean cosine to the k-means (k=4) HCC bank |

Model: `EfficientNetV2B0 + CE + SupCon`, 384×384 input, 12×12 patch grid, `d_model = 128`.
Weights: `cosine_probe_result/effnet_supcon/classifier_y0taaeki_BM.weights.h5`.
Cut-offs are the validation Youden points from `cosine_probe_result/effnet_supcon/cutoffs.csv`.

> Research and debugging tool. Not a medical device. Never use it for diagnosis or treatment.

## 1. Phone-photo cleanup (optional, on by default, every stage switchable)

The model was trained on clean, effectively grayscale SMC-LUD frames (measured mean channel spread
across `example_images_hcc_case/`: 0.0–8.5 of 255). A phone photo is not that, so five corrections run
in this fixed order, and each one can be toggled while a frame is on screen:

1. `remove_glare` — blown-out specular pixels are masked, opened (a 1-px-thin bright structure is
   anatomy, not glare), grown back by one pixel and filled from the local mean of the surviving
   pixels, then diffused. Runs first on purpose: illumination flattening rescales blown pixels back
   into range and would hide them from the detector.
2. `flatten_illumination` — the large-scale brightness ramp from an unlit screen corner or vignette.
3. `desaturate` — collapses the photo's colour cast to the grayscale training domain.
4. `denoise` — 5×5 bilateral filter: sensor grain and JPEG ringing, with lesion margins preserved.
5. `stretch_contrast` — 1st–99th percentile stretch, undoing exposure differences.

The panel reports how many stages ran and what fraction of the frame the glare mask covered. The
viewer has a **RAW / CLEANED** toggle with an opacity slider, so what was removed is always visible.

Reference implementation: `tools/photo_pipeline.py`. The Kotlin port is `ml/PhotoPreprocess.kt` and
`PhotoPreprocessTest` pins both to the same fixture.

## 2. Two resamplers, because the repository has two

The repository preprocesses differently in two places, and they disagree:

| Mode | Mirrors | Notes |
|---|---|---|
| `BILINEAR_UINT8` (default) | `dataloader.py::_decode_image` — `tf.image.resize(bilinear)` then `tf.cast(uint8)` | the pipeline the reported validation/test metrics came from |
| `AREA` | `inference/explainable_predictor.py` — `cv2.resize(INTER_AREA)` | what the demo/visualisation figures use |

Both are fixture-verified against their Python counterpart, and the app labels which one is active.

## 3. Export the model (once, on a machine with TensorFlow)

```bash
python android_app/tools/export_tflite.py
```

Stages into `app/src/main/model/` (git-ignored):

- `hcc_classifier_fp32.tflite` (27.2 MB) and `hcc_classifier_fp16.tflite` (13.8 MB)
- `model_meta_fp32.json` / `model_meta_fp16.json` — each self-consistent, with its own measured drift
- `proto_hcc_mean.bin`, `proto_hem_mean.bin`, `proto_hcc_kmeans.bin`
- `app/src/test/resources/python_reference.json` — ground truth for every unit test

The script verifies each flatbuffer against Keras on four example images and prints the worst
`Δcosine` / `Δprobability`. **fp16 is measured at Δcosine = 9.5e-3**, which is why fp32 is the default
and the app prints `fp16 · Δcos ≤ 9.5e-3` in warning colour when the slim model is packaged.

Grad-CAM is **not** in the app: TFLite has no gradient support. The Python predictor keeps it; the app
shows CA attention rollout and HCC prototype similarity instead.

## 4. Build

```bash
cd android_app
./gradlew :app:assembleDebug                     # fp32, exact      → app-debug.apk
./gradlew :app:assembleRelease \
  -PmodelPrecision=fp16                          # fp16, minified,  → app-release.apk
```

`-PmodelPrecision` picks which staged flatbuffer is packaged (`fp32` default). The release build is
shrunk, resource-shrunk, arm64-only and signed with the debug key so it installs for debugging.

Toolchain used and verified: JDK 17 (Temurin), Gradle 8.11.1, AGP 8.7.3, Kotlin 2.0.21, CameraX
1.4.1, compileSdk 35, minSdk 26, TFLite 2.16.1. `tools/setup-android-toolchain.sh` installs the JDK,
SDK and Gradle into `$HOME/android-dev` without root.

## 5. Install

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

The debug variant gets `applicationIdSuffix = ".debug"`, so it installs next to a release build.
`CAMERA` is the only permission requested (in-app capture); there is deliberately no `INTERNET`
permission and the capture file is deleted as soon as the frame is decoded.

## 6. Tests

```bash
./gradlew :app:testDebugUnitTest
```

17 tests, all green:

- `ScoreMathTest` (6) — area-average resize vs a float64 reference *and* `cv2.INTER_AREA`, cosine
  scores on four real frames, prototype-map formula (synthetic + real patch grid), attention
  normalisation.
- `ResamplerTest` (3) — `tf.image.resize` + `uint8` reproduced exactly, `cv2.INTER_AREA` reproduced,
  and a guard proving the two modes actually disagree (a mode swap cannot pass silently).
- `PhotoPreprocessTest` (4) — all three option sets of the cleanup reproduced against numpy including
  the glare fraction, the disabled pass proven to be the identity, glare detection bounded, and the
  stage order pinned.
- `ColormapTest` (4) — ramps, clamping, linear blend.

## 7. What was verified, and what was not

Verified on 2026-10-18:

- fp32 flatbuffer matches Keras to `Δcosine = 3.9e-06` / `Δprobability = 7.9e-08`; fp16 was measured
  at `Δcosine = 9.5e-03` and therefore rejected as the default.
- `tf.image.resize(bilinear) + uint8` matches `tf.image.resize` exactly (0.0 difference) and
  `cv2.INTER_AREA` matches the float64 area reference to 0.499 (pure uint8 rounding).
- 17/17 unit tests, `:app:assembleDebug` and `:app:assembleRelease` both build.
- The merged manifest resolves the camera permission and `minSdk 26` uses no unguarded newer API.
- A reviewer pass over the whole app produced 13 findings (crash, concurrency, storage, doc drift);
  all acted on. A second look at the cleanup pipeline caught two real defects: illumination flattening
  was hiding the glare (fixed by reordering) and 96 diffusion sweeps cannot fill a large blob (fixed
  by seeding the interior from the local mean of surviving pixels first).

Not verified (no device or emulator on the build machine):

- CameraX preview framing, `ImageCapture` output and runtime permission flow on a real device.
- On-device inference latency and TFLite XNNPACK behaviour on ARM.
- Visual layout on a physical screen. `page-check.sh` / `shot.sh` from the design skill are web-only,
  so they do not apply to Compose; the layout was reviewed against `docs/visual-direction.md` from the
  code and the geometry/contrast numbers were not measured on a screen.
- The cleanup parameters were tuned on a synthetic frame plus reasoning about the training domain —
  **not** on a set of real phone photos. The glare thresholds (`238`, local excess `45` above a
  `215` floor), the bilateral sigma (`18`) and the percentile band are the first guesses, and the
  toggles exist precisely so they can be re-tuned per device.

Known fidelity gaps:

- Decoding and resampling on-device is not bit-identical to Python's `cv2.imread` path; expect drift
  in the third decimal. The resample *algorithms* are verified equal.
- `fp16` drifts the cosine markers by up to 9.5e-3 — fine for eyeballing, not for reproducing a
  reported number.
- Alpha-bearing PNGs lose transparency and read as black, matching `cv2.imread` behaviour.

Performance characteristics:

- The panel prints the cleanup + resample time and the inference time separately. The cleanup is the
  heavier of the two: full-frame box integrals plus a 25-tap bilateral, and it allocates a few
  float64 frame buffers per run (~40 MB of churn at 1000×750). It all runs off the main thread.
- If a low-end device needs the time back, disabling `denoise` or `flatten_illumination` is the
  cheapest win; `remove_glare` and `desaturate` are near-free.

## 8. Design

`docs/project-summary.md` (scope and non-goals), `docs/ux-flow.md` (states and edges),
`docs/visual-direction.md` (palette, type, motion, AI-slop guardrails). Fonts: Space Grotesk
(OFL-1.1, Latin subset) — licence text at `app/src/main/res/raw/space_grotesk_ofl.txt`.
