"""Why does the same frame read differently in the app/browser than in the reference run?

Checks EXIF orientation (both the browser and the app rotate by it, the reference tool does not),
augmentation sensitivity (rotations/flips), and where the field of view flips the label.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import photo_pipeline  # noqa: E402
from export_tflite import CUTOFFS, IMAGE_SIZE, PROBE_DIR, cosine_to, run_tflite  # noqa: E402

MODEL = PROJECT_ROOT / "android_app" / "app" / "src" / "main" / "model" / "hcc_classifier_fp32.tflite"
ORIENTATION = 274


def score(model_bytes, frame, hcc_mean, hem_mean) -> tuple[float, float, float]:
    resized = photo_pipeline.bilinear_uint8(frame.astype(np.float32), IMAGE_SIZE, IMAGE_SIZE)
    outputs = run_tflite(model_bytes, resized[None].astype(np.float32))
    embedding = outputs["embedding"][0]
    hcc = cosine_to(embedding, hcc_mean)
    hem = cosine_to(embedding, hem_mean)
    return float(outputs["probs"][0][1]), hcc, hcc - hem


def center_crop(frame: np.ndarray, fraction: float) -> np.ndarray:
    height, width = frame.shape[:2]
    new_h, new_w = int(height * fraction), int(width * fraction)
    top, left = (height - new_h) // 2, (width - new_w) // 2
    return frame[top : top + new_h, left : left + new_w]


def main() -> None:
    model_bytes = MODEL.read_bytes()
    hcc_mean = np.load(PROBE_DIR / "hcc_bank_mean.npy").astype(np.float32)[0]
    hem_mean = np.load(PROBE_DIR / "hem_bank_mean.npy").astype(np.float32)[0]

    for argument in sys.argv[1:]:
        path = Path(argument)
        with Image.open(path) as image:
            orientation = image.getexif().get(ORIENTATION)
            frame = np.array(image.convert("RGB"))
        print(f"\n=== {path.name}  size={frame.shape[1]}x{frame.shape[0]}  EXIF orientation={orientation}")

        variants = {
            "as-is": frame,
            "rot90": np.rot90(frame, 1),
            "rot180": np.rot90(frame, 2),
            "rot270": np.rot90(frame, 3),
            "flipH": frame[:, ::-1],
            "flipV": frame[::-1, :],
        }
        for label, variant in variants.items():
            probability, hcc, delta = score(model_bytes, variant, hcc_mean, hem_mean)
            verdict = "HCC" if (probability >= CUTOFFS["confidence_score"]) and (delta >= CUTOFFS["delta_score"]) else "Hemangioma"
            print(f"  {label:10s} P(HCC)={probability:8.5f} hccCos={hcc:+.4f} dscore={delta:+.4f} -> {verdict}")

        print("  --- field of view sweep (centre crop) ---")
        for fraction in (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50):
            probability, hcc, delta = score(model_bytes, center_crop(frame, fraction), hcc_mean, hem_mean)
            verdict = "HCC" if (probability >= CUTOFFS["confidence_score"]) and (delta >= CUTOFFS["delta_score"]) else "Hemangioma"
            print(f"  crop {fraction:.2f}   P(HCC)={probability:8.5f} hccCos={hcc:+.4f} dscore={delta:+.4f} -> {verdict}")


if __name__ == "__main__":
    main()
