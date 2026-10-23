"""Diagnose why a frame is labelled HCC: vary preprocessing, crop and correction, and read the maps.

`P(HCC)` alone does not say *why*. This tool runs the same TFLite flatbuffer the app ships over a
matrix of input variants and reports, for each, the four markers plus two spatial diagnostics:

  - attention centre of mass (0.5/0.5 = dead centre). Mass drifting to a corner is the signature of
    the model looking at machine UI, an annotation or a caliper instead of the lesion.
  - prototype-map centre vs border, i.e. does the high-similarity region sit where the lesion is.

Usage:
    python android_app/tools/diagnose_images.py <image> [<image> ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

import photo_pipeline  # noqa: E402
from export_tflite import CUTOFFS, IMAGE_SIZE, PROBE_DIR, area_average_float, cosine_to, prototype_map, run_tflite  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "android_app" / "app" / "src" / "main" / "model" / "hcc_classifier_fp32.tflite"

VARIANTS = (
    ("raw + bilinear_uint8", "bilinear_uint8", 1.0, False),
    ("raw   area (predictor.py)", "area", 1.0, False),
    ("cleanup + bilinear", "bilinear_uint8", 1.0, True),
    ("center crop 0.80", "bilinear_uint8", 0.80, False),
    ("center crop 0.65", "bilinear_uint8", 0.65, False),
)


def load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)


def center_crop(rgb: np.ndarray, fraction: float) -> np.ndarray:
    height, width = rgb.shape[:2]
    new_h, new_w = int(height * fraction), int(width * fraction)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    return rgb[top : top + new_h, left : left + new_w]


def prepare(rgb: np.ndarray, mode: str, fraction: float, cleanup: bool) -> np.ndarray:
    frame = center_crop(rgb, fraction) if fraction < 1.0 else rgb
    if cleanup:
        frame, _ = photo_pipeline.run(frame)
    if mode == "area":
        resized = area_average_float(frame.astype(np.uint8), frame.shape[1], frame.shape[0], IMAGE_SIZE, IMAGE_SIZE)
    else:
        resized = photo_pipeline.bilinear_uint8(frame, IMAGE_SIZE, IMAGE_SIZE)
    return resized[None].astype(np.float32)


def spatial_stats(outputs: dict) -> tuple[float, float, float]:
    grid = outputs["att_map"].shape[1]
    attention = outputs["att_map"][0, :, :, 0].astype(np.float64)
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    rows, cols = np.mgrid[0:grid, 0:grid]
    total = attention.sum()
    centre_x = float((attention * cols).sum() / total / (grid - 1))
    centre_y = float((attention * rows).sum() / total / (grid - 1))

    heat = outputs["prototype_map"][0].reshape(grid, grid)
    mid = slice(grid // 2 - 2, grid // 2 + 2)
    centre_mean = float(heat[mid, mid].mean())
    border = np.concatenate([heat[0, :], heat[-1, :], heat[:, 0], heat[:, -1]])
    return centre_x, centre_y, centre_mean - float(border.mean())


def describe(rgb: np.ndarray) -> str:
    height, width = rgb.shape[:2]
    spread = rgb.max(axis=2) - rgb.min(axis=2)
    luma = photo_pipeline.luma(rgb)
    clipped = float((luma >= 250).mean())
    dark = float((luma <= 5).mean())
    return (
        f"{width}x{height} (aspect {width / height:.2f}) | grayness mean={spread.mean():.1f} p99={np.percentile(spread, 99):.1f}"
        f" | luma mean={luma.mean():.1f} std={luma.std():.1f} | clipped>250={clipped:.3%} near-black={dark:.1%}"
    )


def main() -> None:
    paths = [Path(arg) for arg in sys.argv[1:]]
    if not paths:
        raise SystemExit("usage: python diagnose_images.py <image> [...]")

    model_bytes = MODEL_PATH.read_bytes()
    hcc_mean = np.load(PROBE_DIR / "hcc_bank_mean.npy").astype(np.float32)[0]
    hem_mean = np.load(PROBE_DIR / "hem_bank_mean.npy").astype(np.float32)[0]
    hcc_kmeans = np.load(PROBE_DIR / "hcc_bank_kmeans.npy").astype(np.float32)

    for path in paths:
        rgb = load_rgb(path)
        print(f"\n=== {path.name}")
        print(f"[input] {describe(rgb)}")
        for label, mode, fraction, cleanup in VARIANTS:
            model_input = prepare(rgb, mode, fraction, cleanup)
            outputs = run_tflite(model_bytes, model_input)
            outputs["prototype_map"] = prototype_map(outputs["patch_tokens"][0], hcc_kmeans)[None, ...]
            embedding = outputs["embedding"][0]
            hcc_cosine = cosine_to(embedding, hcc_mean)
            hem_cosine = cosine_to(embedding, hem_mean)
            delta = hcc_cosine - hem_cosine
            probability = float(outputs["probs"][0][1])
            centre_x, centre_y, centre_bonus = spatial_stats(outputs)
            verdicts = [
                probability >= CUTOFFS["confidence_score"],
                hcc_cosine >= CUTOFFS["hcc_cosine_score"],
                delta >= CUTOFFS["delta_score"],
            ]
            consensus = "HCC" if sum(verdicts) >= 2 else "Hemangioma"
            print(
                f"  {label:26s} P(HCC)={probability:7.5f} hccCos={hcc_cosine:+.4f} hemCos={hem_cosine:+.4f}"
                f" dscore={delta:+.4f} -> {consensus:10s} [{sum(verdicts)}/3]"
                f" | attn com=({centre_x:.2f},{centre_y:.2f}) proto centre-border={centre_bonus:+.3f}"
            )


if __name__ == "__main__":
    main()
