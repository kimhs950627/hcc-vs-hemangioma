"""Export the HCC-vs-Hemangioma classifier to TensorFlow Lite for the Android app.

Pipeline
--------
1. Load the primary model (EfficientNetV2B0 + CE + SupCon) and its prototypes.
2. Wrap the encoder + classifier head in a single-input / multi-output signature:
     image  [1, 384, 384, 3] float32 in [0, 255]
       -> probs        [1, 2]          softmax over (hemangioma, HCC)
       -> embedding    [1, 128]        CLS token after final LayerNorm
       -> patch_tokens [1, 144, 128]   patch tokens (12x12 grid) for prototype heatmap
       -> att_map      [1, 12, 12, 1]  cross-attention rollout over the 4 encoder blocks
3. Convert to a float32 TFLite flatbuffer (dynamic-range quantisation is NOT used: it
   shifts the 128-d embedding by up to 2.8e-1 and therefore the cosine scores).
4. Write Android assets: the .tflite plus prototype banks and a meta json.
5. Verify the flatbuffer against Keras on every example image and write a reference
   fixture that the Kotlin unit test asserts against.

Run from the repository root with a TensorFlow-enabled interpreter:

    python android_app/tools/export_tflite.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf  # noqa: E402
from inference.explainable_predictor import ExplainableHCCPredictor  # noqa: E402

# Sibling module holding the numpy reference for the on-device photo pipeline.
import photo_pipeline  # noqa: E402

IMAGE_SIZE = 384
PATCH_GRID = 12
N_PATCHES = PATCH_GRID * PATCH_GRID
EMBED_DIM = 128
PROBE_DIR = PROJECT_ROOT / "cosine_probe_result" / "effnet_supcon"
WEIGHTS_PATH = PROBE_DIR / "classifier_y0taaeki_BM.weights.h5"
# Staged models, not assets: `app/build.gradle.kts` copies the precision selected by
# -PmodelPrecision into a generated asset directory, so one export can serve both builds.
MODEL_DIR = PROJECT_ROOT / "android_app" / "app" / "src" / "main" / "model"
TEST_RESOURCES_DIR = PROJECT_ROOT / "android_app" / "app" / "src" / "test" / "resources"
MODEL_ASSET_NAME = "hcc_classifier.tflite"
FINAL_MODEL_NAME = "hcc_classifier.tflite"
FINAL_META_NAME = "model_meta.json"

# fp16 keeps full float32 activations and only stores weights in half precision. It is accepted only
# if the cosine markers stay within this much of the Keras reference; otherwise fp32 ships instead.
FP16_COSINE_TOLERANCE = 2e-3
FP16_PROBABILITY_TOLERANCE = 2e-3

# Youden's J operating points from the validation split (cosine_probe_result/effnet_supcon/cutoffs.csv)
CUTOFFS = {
    "confidence_score": 0.0010883796494454145,
    "hcc_cosine_score": -0.004843093454837799,
    "delta_score": -0.9584707617759705,
}

EXAMPLE_IMAGES = [
    "example_images_hcc_case/hcc1.jpg",
    "example_images_hcc_case/hcc2.jpg",
    "example_images_hcc_case/hcc5.jpg",
    "example_images_hcc_case/new_hcc.jpg",
]


def forward(encoder, classifier_head, image: tf.Tensor) -> dict[str, tf.Tensor]:
    """Single forward pass returning every quantity the app needs (eager or graph)."""
    enc = encoder(image, training=False, return_attention=True)
    embedding = enc["embedding"]
    probs = tf.nn.softmax(classifier_head(embedding, training=False), axis=-1)
    att_map = _attention_rollout(enc["attention_weights"])
    return {
        "probs": probs,
        "embedding": embedding,
        "patch_tokens": enc["patch_tokens"],
        "att_map": att_map,
    }


def build_bilinear_fixture() -> dict:
    """Same deterministic image through the dataloader's resampler, as Kotlin ground truth."""
    src_width, src_height = 37, 23
    dst_width, dst_height = 12, 8
    rng = np.random.default_rng(20261018)
    pixels = rng.integers(0, 256, size=(src_height, src_width, 3), dtype=np.uint8)

    reference = photo_pipeline.bilinear_uint8(pixels, dst_height, dst_width)
    tf_result = tf.cast(
        tf.image.resize(tf.constant(pixels, dtype=tf.float32), (dst_height, dst_width), method="bilinear"),
        tf.uint8,
    ).numpy()
    print(
        "[fixture] bilinear+uint8 vs tf.image.resize: max diff %.3f (must be 0)"
        % float(np.max(np.abs(reference - tf_result.astype(np.float64))))
    )
    return {
        "src_width": src_width,
        "src_height": src_height,
        "dst_width": dst_width,
        "dst_height": dst_height,
        "src_pixels_rgb": pixels.reshape(-1).astype(int).tolist(),
        "expected_rgb": reference.reshape(-1).tolist(),
        "tolerance": 1e-6,
    }


def build_synthetic_photo(height: int = 36, width: int = 28) -> np.ndarray:
    """A frame with every defect the pipeline targets: backlight ramp, colour cast, glare, grain."""
    rng = np.random.default_rng(20261018)
    rows, cols = np.mgrid[0:height, 0:width]
    backlight = 0.55 + 0.45 * np.clip(1.0 - (cols / width) * 0.9, 0.0, 1.0)
    tissue = 42.0 + 150.0 * (0.5 + 0.5 * np.sin(cols / 5.0) * np.cos(rows / 6.0))
    grain = rng.normal(0.0, 6.0, size=(height, width))
    glare = 255.0 * np.exp(-(((cols - 9.0) ** 2 + (rows - 8.0) ** 2) / (2 * 3.5 ** 2)))
    gray = np.clip(tissue * backlight + grain + glare, 0.0, 255.0)

    cast = np.stack([gray * 1.14, gray * 0.97, gray * 0.84], axis=-1)
    return np.clip(cast, 0.0, 255.0).astype(np.float32)


def build_preprocess_fixture() -> dict:
    """Pipeline output for three option sets, so the Kotlin port and its toggles are both pinned."""
    photo = build_synthetic_photo()
    option_sets = {
        "all_enabled": {},
        "glare_only": dict(flatten=False, denoise_noise=False, contrast=False),
        "disabled": dict(flatten=False, desaturate_channels=False, remove_glare=False, denoise_noise=False, contrast=False),
    }
    cases = {}
    for name, overrides in option_sets.items():
        result, glare_fraction = photo_pipeline.run(photo, **overrides)
        cases[name] = {
            "options": {
                "flatten_illumination": overrides.get("flatten", True),
                "desaturate": overrides.get("desaturate_channels", True),
                "remove_glare": overrides.get("remove_glare", True),
                "denoise": overrides.get("denoise_noise", True),
                "stretch_contrast": overrides.get("contrast", True),
            },
            "expected_rgb": result.reshape(-1).astype(float).tolist(),
            "expected_glare_fraction": glare_fraction,
        }
        print(
            "[fixture] preprocess %-13s glare=%.4f  mean=%.1f -> %.1f"
            % (name, glare_fraction, photo.mean(), result.mean())
        )
    return {
        "height": int(photo.shape[0]),
        "width": int(photo.shape[1]),
        "src_rgb": photo.reshape(-1).astype(float).tolist(),
        "cases": cases,
        "tolerance": 2e-2,
    }


def evaluate(model_bytes: bytes, examples: list[tuple[str, np.ndarray, dict]], hcc_mean: np.ndarray,
             hem_mean: np.ndarray, hcc_kmeans: np.ndarray) -> tuple[dict, list[dict]]:
    """Runs the flatbuffer on every example and measures it against the Keras reference."""
    aggregate = {"probability": 0.0, "embedding": 0.0, "cosine": 0.0}
    cases = []
    for name, image, keras_out in examples:
        lite_out = run_tflite(model_bytes, image)
        deltas = {
            key: float(np.max(np.abs(keras_out[key] - lite_out[key])))
            for key in ("probs", "embedding", "att_map")
        }
        embedding = lite_out["embedding"][0]
        lite_hcc = cosine_to(embedding, hcc_mean[0])
        lite_hem = cosine_to(embedding, hem_mean[0])
        keras_embedding = keras_out["embedding"][0]
        keras_hcc = cosine_to(keras_embedding, hcc_mean[0])
        keras_hem = cosine_to(keras_embedding, hem_mean[0])
        cosine_error = max(abs(lite_hcc - keras_hcc), abs(lite_hem - keras_hem),
                           abs((lite_hcc - lite_hem) - (keras_hcc - keras_hem)))
        aggregate["probability"] = max(aggregate["probability"], deltas["probs"])
        aggregate["embedding"] = max(aggregate["embedding"], deltas["embedding"])
        aggregate["cosine"] = max(aggregate["cosine"], cosine_error)
        print(
            "[verify] %-34s probs=%.2e embedding=%.2e attn=%.2e dcos=%.2e"
            % (name, deltas["probs"], deltas["embedding"], deltas["att_map"], cosine_error)
        )

        case = {
            "name": Path(name).name,
            "probs": [float(v) for v in lite_out["probs"][0]],
            "embedding": [float(v) for v in embedding],
            "hcc_cosine_score": lite_hcc,
            "hem_cosine_score": lite_hem,
            "delta_score": lite_hcc - lite_hem,
            "att_map_raw": [float(v) for v in lite_out["att_map"][0, :, :, 0].ravel()],
            "att_map": [float(v) for v in normalize(lite_out["att_map"][0, :, :, 0]).ravel()],
            "hcc_prototype_map": [float(v) for v in prototype_map(lite_out["patch_tokens"][0], hcc_kmeans).ravel()],
        }
        if not cases:
            # Only the first case ships its patch tokens (18k floats); `prototype_map_case` covers the
            # same formula at a fraction of the size.
            case["patch_tokens"] = [float(v) for v in lite_out["patch_tokens"][0].ravel()]
        cases.append(case)
    return aggregate, cases


def build_serving_fn(encoder, classifier_head) -> tf.types.experimental.GenericFunction:
    """Trace `forward` into a single-input / multi-output TFLite-friendly signature."""

    @tf.function(input_signature=[tf.TensorSpec([1, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32, name="image")])
    def serving(image):
        return forward(encoder, classifier_head, image)

    return serving


def _attention_rollout(attention_weights: list) -> tf.Tensor:
    """Head-mean, layer-weighted attention rollout over CA maps [B, H, 1, N] -> [B, grid, grid, 1]."""
    per_layer = []
    for layer_attn in attention_weights:
        head_mean = tf.reduce_mean(layer_attn, axis=1)[:, 0, :]  # [B, N]
        per_layer.append(head_mean / (tf.reduce_sum(head_mean, axis=-1, keepdims=True) + 1e-8))

    accumulated = per_layer[0]
    for index, layer_map in enumerate(per_layer[1:], start=1):
        weight = index / (index + 1.0)
        accumulated = weight * accumulated + (1.0 - weight) * layer_map
    return tf.reshape(accumulated, [1, PATCH_GRID, PATCH_GRID, 1])


def normalize(field: np.ndarray) -> np.ndarray:
    """Min-max normalise a 12x12 map into [0, 1] for display."""
    low = float(np.min(field))
    high = float(np.max(field))
    return (field - low) / (high - low + 1e-8)


def prototype_map(patch_tokens: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Per-patch mean cosine similarity to a prototype bank, reshaped to the patch grid."""
    tokens = patch_tokens / (np.linalg.norm(patch_tokens, axis=-1, keepdims=True) + 1e-8)
    protos = prototypes / (np.linalg.norm(prototypes, axis=-1, keepdims=True) + 1e-8)
    per_patch = (tokens @ protos.T).mean(axis=-1)  # [144]
    return normalize(per_patch.reshape(PATCH_GRID, PATCH_GRID))


def cosine_to(embedding: np.ndarray, prototype: np.ndarray) -> float:
    embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-8)
    prototype_norm = prototype / (np.linalg.norm(prototype) + 1e-8)
    return float(np.dot(embedding_norm, prototype_norm))


def load_image(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)[None]


def map_lite_outputs(interpreter: tf.lite.Interpreter) -> dict[str, np.ndarray]:
    """TFLite output order is not stable; key each tensor by its shape."""
    expected = {
        (1, PATCH_GRID, PATCH_GRID, 1): "att_map",
        (1, EMBED_DIM): "embedding",
        (1, N_PATCHES, EMBED_DIM): "patch_tokens",
        (1, 2): "probs",
    }
    mapped = {}
    for detail in interpreter.get_output_details():
        shape = tuple(int(d) for d in detail["shape"])
        if shape not in expected:
            raise RuntimeError(f"unexpected TFLite output shape: {shape}")
        mapped[expected[shape]] = interpreter.get_tensor(detail["index"])
    missing = set(expected.values()) - set(mapped)
    if missing:
        raise RuntimeError(f"missing TFLite outputs: {sorted(missing)}")
    return mapped


def run_tflite(model_bytes: bytes, image: np.ndarray) -> dict[str, np.ndarray]:
    interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=4)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]["index"], image)
    interpreter.invoke()
    return map_lite_outputs(interpreter)


def build_prototype_map_fixture() -> dict:
    """Small deterministic case proving ScoreMath.prototypeMap against the numpy formula."""
    grid, dim, bank = 4, 8, 3
    tokens = np.arange(grid * grid * dim, dtype=np.float32).reshape(grid * grid, dim) / 37.0 - 4.0
    prototypes = (
        np.arange(bank * dim, dtype=np.float32).reshape(bank, dim) / 11.0 - 2.0
    )
    tokens_n = tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)
    protos_n = prototypes / (np.linalg.norm(prototypes, axis=-1, keepdims=True) + 1e-8)
    per_patch = (tokens_n @ protos_n.T).mean(axis=-1)
    expected = normalize(per_patch.reshape(grid, grid))
    return {
        "grid": grid,
        "dim": dim,
        "patch_tokens": tokens.reshape(-1).tolist(),
        "prototypes": prototypes.tolist(),
        "expected": expected.reshape(-1).tolist(),
        "tolerance": 1e-5,
    }


def area_average_float(pixels: np.ndarray, src_w: int, src_h: int, dst_w: int, dst_h: int) -> np.ndarray:
    """Reference implementation of the coverage-weighted downscale, in float64 (no rounding)."""
    out = np.zeros((dst_h, dst_w, 3), dtype=np.float64)
    source = pixels.astype(np.float64)
    scale_x = src_w / dst_w
    scale_y = src_h / dst_h
    for dy in range(dst_h):
        top, bottom = dy * scale_y, (dy + 1) * scale_y
        for dx in range(dst_w):
            left, right = dx * scale_x, (dx + 1) * scale_x
            accum = np.zeros(3, dtype=np.float64)
            weight_sum = 0.0
            for y in range(int(np.floor(top)), min(int(np.ceil(bottom)), src_h)):
                weight_y = max(0.0, min(y + 1.0, bottom) - max(float(y), top))
                if weight_y <= 0.0:
                    continue
                for x in range(int(np.floor(left)), min(int(np.ceil(right)), src_w)):
                    weight_x = max(0.0, min(x + 1.0, right) - max(float(x), left))
                    if weight_x <= 0.0:
                        continue
                    weight = weight_x * weight_y
                    accum += source[y, x] * weight
                    weight_sum += weight
            out[dy, dx] = accum / weight_sum if weight_sum > 0.0 else 0.0
    return out


def build_resize_fixture() -> dict:
    """Deterministic RGB image resized with cv2.INTER_AREA, as ground truth for the Kotlin resizer."""
    import cv2

    src_width, src_height = 37, 23
    dst_width, dst_height = 12, 8
    rng = np.random.default_rng(20261018)
    pixels = rng.integers(0, 256, size=(src_height, src_width, 3), dtype=np.uint8)
    opencv_result = cv2.resize(pixels, (dst_width, dst_height), interpolation=cv2.INTER_AREA)
    float_result = area_average_float(pixels, src_width, src_height, dst_width, dst_height)
    print(
        "[fixture] cv2.INTER_AREA vs float64 area average: max diff %.3f (uint8 rounding only)"
        % float(np.max(np.abs(opencv_result.astype(np.float64) - float_result)))
    )
    return {
        "src_width": src_width,
        "src_height": src_height,
        "dst_width": dst_width,
        "dst_height": dst_height,
        "src_pixels_rgb": pixels.reshape(-1).astype(int).tolist(),
        "expected_rgb": float_result.reshape(-1).tolist(),
        "opencv_rounded_rgb": opencv_result.astype(float).reshape(-1).tolist(),
        "opencv_rounding_tolerance": 0.5,
        # The Kotlin side returns float32 (the model input dtype), so the float64 reference can only
        # be matched to float32 resolution: ulp(255) = 1.5e-5, so 1e-3 is a safe bound.
        "tolerance": 1e-3,
    }


def write_prototype(assets_dir: Path, name: str, array: np.ndarray) -> None:
    (assets_dir / name).write_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def write_meta(path: Path, precision: str, model_bytes: int, report: dict) -> None:
    """Writes one self-consistent meta file per precision, staged next to the flatbuffers."""
    meta = {
        "model_asset": FINAL_MODEL_NAME,
        "model_bytes": model_bytes,
        "precision": precision,
        "precision_drift": {
            "probability": report["probability"],
            "embedding": report["embedding"],
            "cosine": report["cosine"],
        },
        "input": {"height": IMAGE_SIZE, "width": IMAGE_SIZE, "value_range": [0.0, 255.0]},
        "labels": ["Hemangioma", "HCC"],
        "positive_label": "HCC",
        "patch_grid": PATCH_GRID,
        "embed_dim": EMBED_DIM,
        "prototypes": {
            "hcc_mean": "proto_hcc_mean.bin",
            "hem_mean": "proto_hem_mean.bin",
            "hcc_kmeans": "proto_hcc_kmeans.bin",
        },
        "cutoffs": CUTOFFS,
        "training": "EfficientNetV2B0 + CE + SupCon",
        "resize_modes": {
            "bilinear_uint8": "tf.image.resize(bilinear) + uint8 cast — dataloader.py, the pipeline the reported metrics came from",
            "area": "area average — inference/explainable_predictor.py, the demo/visualisation pipeline",
        },
        "preprocess_stages": {
            stage: description
            for stage, description in (
                (photo_pipeline.STAGE_GLARE, "blown-out specular pixels masked (opened to spare thin anatomy), filled from the surrounding tissue"),
                (photo_pipeline.STAGE_FLATTEN, "large-scale brightness ramp (backlight, vignette) removed"),
                (photo_pipeline.STAGE_DESATURATE, "collapse to luma: the training frames are effectively grayscale"),
                (photo_pipeline.STAGE_DENOISE, "5x5 bilateral filter: sensor noise and JPEG ringing"),
                (photo_pipeline.STAGE_CONTRAST, "1-99 percentile stretch: exposure normalisation"),
            )
        },
        "preprocess_order": list(photo_pipeline.STAGE_ORDER),
        "drop_gradcam_reason": "Grad-CAM needs gradients; TFLite has none. The app shows CA attention rollout and HCC prototype similarity instead.",
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    started = time.time()
    predictor = ExplainableHCCPredictor(weights_path=WEIGHTS_PATH, probe_dir=PROBE_DIR)

    hcc_mean = np.load(PROBE_DIR / "hcc_bank_mean.npy").astype(np.float32)  # [1, 128]
    hem_mean = np.load(PROBE_DIR / "hem_bank_mean.npy").astype(np.float32)
    hcc_kmeans = np.load(PROBE_DIR / "hcc_bank_kmeans.npy").astype(np.float32)  # [4, 128]

    serving_fn = build_serving_fn(predictor.encoder, predictor.model.classifier_head)
    concrete = serving_fn.get_concrete_function()

    fp32_bytes = tf.lite.TFLiteConverter.from_concrete_functions([concrete], serving_fn).convert()
    print(f"[export] fp32 flatbuffer: {len(fp32_bytes):,} bytes in {time.time() - started:.1f}s")

    fp16_bytes = None
    try:
        fp16_converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], serving_fn)
        fp16_converter.optimizations = [tf.lite.Optimize.DEFAULT]
        fp16_converter.target_spec.supported_types = [tf.float16]
        fp16_bytes = fp16_converter.convert()
        print(f"[export] fp16 flatbuffer: {len(fp16_bytes):,} bytes")
    except Exception as error:  # noqa: BLE001 - any conversion failure just means fp32 ships
        print(f"[export] fp16 conversion unavailable: {error}")

    examples = []
    for relative in EXAMPLE_IMAGES:
        image_path = PROJECT_ROOT / relative
        if not image_path.exists():
            print(f"[verify] skip missing {relative}")
            continue
        image = load_image(image_path)
        keras_out = forward(predictor.encoder, predictor.model.classifier_head, tf.constant(image))
        examples.append((Path(relative).stem, image, {key: value.numpy() for key, value in keras_out.items()}))

    print("[verify] --- fp32 ---")
    fp32_report, fp32_cases = evaluate(fp32_bytes, examples, hcc_mean, hem_mean, hcc_kmeans)

    model_bytes, precision, report, cases = fp32_bytes, "fp32", fp32_report, fp32_cases
    if fp16_bytes is not None:
        print("[verify] --- fp16 ---")
        fp16_report, fp16_cases = evaluate(fp16_bytes, examples, hcc_mean, hem_mean, hcc_kmeans)
        cosine_ok = fp16_report["cosine"] <= FP16_COSINE_TOLERANCE
        probability_ok = fp16_report["probability"] <= FP16_PROBABILITY_TOLERANCE
        if cosine_ok and probability_ok:
            model_bytes, precision, report, cases = fp16_bytes, "fp16", fp16_report, fp16_cases
        else:
            print(
                "[export] fp16 rejected: dcos=%.2e (limit %.0e), dprob=%.2e (limit %.0e)"
                % (fp16_report["cosine"], FP16_COSINE_TOLERANCE, fp16_report["probability"], FP16_PROBABILITY_TOLERANCE)
            )
    print(
        "[export] shipping %s, %d bytes | worst dcos=%.2e dprob=%.2e demb=%.2e"
        % (precision, len(model_bytes), report["cosine"], report["probability"], report["embedding"])
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "hcc_classifier_fp32.tflite").write_bytes(fp32_bytes)
    if fp16_bytes is not None:
        (MODEL_DIR / "hcc_classifier_fp16.tflite").write_bytes(fp16_bytes)
    write_prototype(MODEL_DIR, "proto_hcc_mean.bin", hcc_mean)
    write_prototype(MODEL_DIR, "proto_hem_mean.bin", hem_mean)
    write_prototype(MODEL_DIR, "proto_hcc_kmeans.bin", hcc_kmeans)
    write_meta(MODEL_DIR / "model_meta_fp32.json", "fp32", len(fp32_bytes), fp32_report)
    if fp16_bytes is not None:
        write_meta(MODEL_DIR / "model_meta_fp16.json", "fp16", len(fp16_bytes), fp16_report)

    TEST_RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {
        "generated_by": "android_app/tools/export_tflite.py",
        "reference": "TFLite interpreter output on example_images_hcc_case, and the numpy reference for every Kotlin numeric stage",
        "precision": precision,
        "max_abs_diff_vs_keras": report,
        "prototypes": {"hcc_mean": hcc_mean.tolist(), "hem_mean": hem_mean.tolist(), "hcc_kmeans": hcc_kmeans.tolist()},
        "cutoffs": CUTOFFS,
        "resize_case": build_resize_fixture(),
        "bilinear_case": build_bilinear_fixture(),
        "preprocess_case": build_preprocess_fixture(),
        "prototype_map_case": build_prototype_map_fixture(),
        "cases": cases,
    }
    (TEST_RESOURCES_DIR / "python_reference.json").write_text(json.dumps(fixture), encoding="utf-8")
    print(f"[export] fixture written ({len(cases)} cases); models staged in {MODEL_DIR}")


if __name__ == "__main__":
    main()
