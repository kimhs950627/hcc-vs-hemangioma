"""Export the classifier to ONNX for the GitHub Pages demo (ONNX Runtime Web).

Why ONNX and not TensorFlow.js: the model's exact GELU lowers to `Erfc`, which tfjs has no kernel for,
and Keras' effort to rewrite it as `erf` keeps getting re-fused by grappler into the `Gelu`
(`geluexact`) op, which tfjs also lacks. ONNX carries `Erfc`/`Gelu` as standard operators, so no
graph surgery is needed at all — the model that ships is the paper's model.

Steps:
1. SavedModel from the same serving signature as the TFLite export (image -> probs, embedding,
   patch_tokens, att_map).
2. `tf2onnx` conversion to a single fp32 `.onnx`.
3. Verification with `onnxruntime` against `examples/expected.json` (which comes from the TFLite fp32
   flatbuffer). That check is the gate: if the ONNX graph does not reproduce the app's numbers, the
   demo is not shipping.

    python android_app/tools/export_web_onnx.py [--skip-convert]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(TOOLS_DIR))

import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from inference.explainable_predictor import ExplainableHCCPredictor  # noqa: E402

from export_tflite import IMAGE_SIZE, PROBE_DIR, WEIGHTS_PATH, cosine_to, forward, prototype_map  # noqa: E402

WEB_DIR = PROJECT_ROOT / "docs" / "web-demo"
EXAMPLE_DIR = WEB_DIR / "examples"
HEMANGIOMA_DIR = PROJECT_ROOT / "example_images_hemangioma_case"
EXAMPLE_JPEG_QUALITY = 95
# Four known HCC frames and four known hemangioma frames, so a visitor sees both outcomes.
DEMO_EXAMPLES = [
    ("example_images_hcc_case/hcc1.jpg", "HCC"),
    ("example_images_hcc_case/hcc2.jpg", "HCC"),
    ("example_images_hcc_case/hcc5.jpg", "HCC"),
    ("example_images_hcc_case/new_hcc.jpg", "HCC"),
    ("example_images_hemangioma_case/hem1.jpg", "Hemangioma"),
    ("example_images_hemangioma_case/hem2.jpg", "Hemangioma"),
    ("example_images_hemangioma_case/hem3.jpg", "Hemangioma"),
    ("example_images_hemangioma_case/hem4.png", "Hemangioma"),
]
MODEL_DIR = WEB_DIR / "model"
ONNX_NAME = "hcc_classifier.onnx"
COSINE_TOLERANCE = 2e-3
PROBABILITY_TOLERANCE = 2e-3
MAP_TOLERANCE = 5e-3


def iter_layers(root, seen=None) -> list:
    """Every Keras layer reachable from `root`, walking instance attributes.

    A subclassed `keras.Model` does not populate `.layers`, so the attribute graph is the only way to
    reach the encoder's `blocks` list and the `ffn` Sequential inside each block.
    """
    if seen is None:
        seen = set()
    if id(root) in seen:
        return []
    seen.add(id(root))

    found = []
    for value in vars(root).values():
        if isinstance(value, keras.layers.Layer):
            candidates = [value]
        elif isinstance(value, (list, tuple)):
            candidates = [item for item in value if isinstance(item, keras.layers.Layer)]
        elif isinstance(value, dict):
            candidates = [item for item in value.values() if isinstance(item, keras.layers.Layer)]
        else:
            candidates = []
        for candidate in candidates:
            found.append(candidate)
            found.extend(iter_layers(candidate, seen))
    return found


def swap_gelu_activations(roots) -> int:
    """Replaces every GELU activation with the `erf` form before saving.

    ONNX has no registered `Erfc` at these opsets (checked at 17 and 15: "No Op registered for
    Erfc"), while `Erf` is standard everywhere. `0.5x(1+erf(x/sqrt2))` is the same function to ~1e-7,
    and the verification below measures the difference rather than assuming it.
    """
    import keras.src.activations as keras_activations

    def erf_gelu(inputs):
        return 0.5 * inputs * (1.0 + tf.math.erf(inputs * 0.7071067811865476))

    candidates = list(roots)
    for root in roots:
        candidates.extend(iter_layers(root))

    swapped = 0
    for layer in candidates:
        if getattr(layer, "activation", None) is keras_activations.gelu:
            layer.activation = erf_gelu
            swapped += 1
    return swapped


class ServingModule(tf.Module):
    """Trackable wrapper so the SavedModel carries the encoder and classifier-head variables."""

    def __init__(self, encoder, classifier_head):
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head

    @tf.function(input_signature=[tf.TensorSpec([1, IMAGE_SIZE, IMAGE_SIZE, 3], tf.float32, name="image")])
    def serve(self, image):
        return forward(self.encoder, self.classifier_head, image)


def save_saved_model(predictor, target: Path) -> None:
    module = ServingModule(predictor.encoder, predictor.model.classifier_head)
    swapped = swap_gelu_activations([predictor.encoder, predictor.model.classifier_head])
    print(f"[savedmodel] GELU -> erf on {swapped} layer(s)")
    assert swapped > 0, "no GELU activation found; the Erfc problem would come back"
    concrete = module.serve.get_concrete_function()
    tf.saved_model.save(module, str(target), signatures={"serving_default": concrete})
    print(f"[savedmodel] wrote {target}")


def convert_to_onnx(saved_model_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "tf2onnx.convert",
        "--saved-model",
        str(saved_model_dir),
        "--output",
        str(output_path),
        "--opset",
        "15",
        "--tag",
        "serve",
        "--signature",
        "serving_default",
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:])
        raise SystemExit(f"tf2onnx failed with exit code {completed.returncode}")
    print(f"[onnx] {output_path.name}: {output_path.stat().st_size / 1e6:.1f} MB")


def write_runtime_names(config_path: Path, session) -> None:
    """Records the ONNX graph's real tensor names so the page never maps by shape or by guess.

    The earlier TensorFlow.js attempt left Identity-style names in the config; ONNX keeps the dict
    keys (`probs`, `embedding`, `patch_tokens`, `att_map`), so the config is rewritten from the graph.
    """
    config = json.loads(config_path.read_text(encoding="utf-8"))
    names = {output.name for output in session.get_outputs()}
    expected = ("probs", "embedding", "patch_tokens", "att_map")
    missing = [name for name in expected if name not in names]
    if missing:
        raise SystemExit(f"ONNX graph is missing expected outputs {missing}; got {sorted(names)}")
    config["outputs"] = {name: name for name in expected}
    config["inputName"] = session.get_inputs()[0].name
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"[config] input={config['inputName']} outputs={list(config['outputs'])}")


def demo_model_input(image_path: Path) -> np.ndarray:
    """The page's own preprocessing: bilinear + uint8 resample, i.e. the dataloader pipeline."""
    import cv2

    from photo_pipeline import bilinear_uint8

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise SystemExit(f"cannot read {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return bilinear_uint8(rgb, IMAGE_SIZE, IMAGE_SIZE)[None].astype(np.float32)


def write_examples() -> list[dict]:
    """Stages the demo frames, their raw model inputs and the manifest.

    Every frame is re-encoded to JPEG at high quality so the page handles one format, and the exact
    384x384x3 tensor is dumped beside it so the page (and its self-test) needs no JPEG decoder.
    """
    import cv2

    EXAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    for relative, label in DEMO_EXAMPLES:
        source = PROJECT_ROOT / relative
        if not source.exists():
            print(f"[examples] skip missing {relative}")
            continue
        frame = cv2.imread(str(source))
        if frame is None:
            print(f"[examples] unreadable {relative}")
            continue
        target = EXAMPLE_DIR / f"{source.stem}.jpg"
        cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, EXAMPLE_JPEG_QUALITY])
        demo_model_input(target).astype(np.float32).tofile(EXAMPLE_DIR / f"{source.stem}.bin")
        entries.append({"file": target.name, "label": label, "source": relative})
    (EXAMPLE_DIR / "manifest.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    print(f"[examples] {len(entries)} frame(s) staged")
    return entries


def _normalise(values: np.ndarray) -> np.ndarray:
    low, high = float(values.min()), float(values.max())
    return (values - low) / (high - low + 1e-8)


def write_expected(manifest: list[dict]) -> None:
    """Reference markers per example, from the TFLite fp32 model the Android app ships."""
    from export_tflite import run_tflite

    model_path = PROJECT_ROOT / "android_app" / "app" / "src" / "main" / "model" / "hcc_classifier_fp32.tflite"
    model_bytes = model_path.read_bytes()
    hcc_mean = np.load(PROBE_DIR / "hcc_bank_mean.npy").astype(np.float32)[0]
    hem_mean = np.load(PROBE_DIR / "hem_bank_mean.npy").astype(np.float32)[0]
    hcc_kmeans = np.load(PROBE_DIR / "hcc_bank_kmeans.npy").astype(np.float32)

    expected = {}
    for entry in manifest:
        outputs = run_tflite(model_bytes, demo_model_input(EXAMPLE_DIR / entry["file"]))
        embedding = outputs["embedding"][0]
        hcc_cosine = cosine_to(embedding, hcc_mean)
        hem_cosine = cosine_to(embedding, hem_mean)
        expected[entry["file"]] = {
            "probs": [float(v) for v in outputs["probs"][0]],
            "hccCosine": hcc_cosine,
            "hemCosine": hem_cosine,
            "delta": hcc_cosine - hem_cosine,
            "attentionMap": [float(v) for v in _normalise(outputs["att_map"][0, :, :, 0]).ravel()],
            "prototypeMap": [float(v) for v in prototype_map(outputs["patch_tokens"][0], hcc_kmeans).ravel()],
        }
        print(
            "[expected] %-10s %-11s P(HCC)=%8.5f HCCcos=%+.4f dscore=%+.4f"
            % (
                entry["file"],
                entry["label"],
                expected[entry["file"]]["probs"][1],
                hcc_cosine,
                hcc_cosine - hem_cosine,
            )
        )
    (EXAMPLE_DIR / "expected.json").write_text(json.dumps(expected), encoding="utf-8")


def verify_onnx(model_path: Path) -> int:
    """Runs every staged example through ONNX Runtime and compares with the app's own numbers."""
    import onnxruntime as ort

    expected = json.loads((WEB_DIR / "examples" / "expected.json").read_text(encoding="utf-8"))
    config = json.loads((WEB_DIR / "config.json").read_text(encoding="utf-8"))
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    outputs = {output.name for output in session.get_outputs()}
    print(f"[verify] onnxruntime {ort.__version__} outputs={sorted(outputs)}")
    write_runtime_names(WEB_DIR / "config.json", session)

    hcc_kmeans = np.array(config["prototypes"]["hccKmeans"], dtype=np.float32)
    hcc_mean = np.array(config["prototypes"]["hccMean"], dtype=np.float32)
    hem_mean = np.array(config["prototypes"]["hemMean"], dtype=np.float32)
    positive_index = config["positiveIndex"]
    grid = config["patchGrid"]
    dim = config["embedDim"]

    failures = 0
    for file, reference in expected.items():
        model_input = np.fromfile(
            WEB_DIR / "examples" / file.replace(".jpg", ".bin"), dtype=np.float32
        ).reshape(1, config["inputSize"], config["inputSize"], 3)

        by_shape = {
            tuple(np.asarray(value).shape): value
            for value in session.run(None, {session.get_inputs()[0].name: model_input})
        }
        probs = by_shape[(1, 2)][0]
        embedding = by_shape[(1, dim)][0]
        patches = by_shape[(1, grid * grid, dim)][0]
        attention = by_shape[(1, grid, grid, 1)][0, :, :, 0]

        hcc_cosine = cosine_to(embedding, hcc_mean)
        hem_cosine = cosine_to(embedding, hem_mean)
        delta = hcc_cosine - hem_cosine
        map_values = prototype_map(patches, hcc_kmeans)
        attention_values = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        probability_delta = abs(float(probs[positive_index]) - reference["probs"][positive_index])
        cosine_delta = max(
            abs(hcc_cosine - reference["hccCosine"]),
            abs(hem_cosine - reference["hemCosine"]),
            abs(delta - reference["delta"]),
        )
        map_delta = float(np.max(np.abs(map_values.ravel() - np.array(reference["prototypeMap"]))))
        attention_delta = float(
            np.max(np.abs(attention_values.ravel() - np.array(reference["attentionMap"])))
        )

        ok = (
            probability_delta <= PROBABILITY_TOLERANCE
            and cosine_delta <= COSINE_TOLERANCE
            and map_delta <= MAP_TOLERANCE
        )
        failures += 0 if ok else 1
        print(
            "[verify] %s %-14s dP=%.2e dCos=%.2e dMap=%.2e dAttn=%.2e | P(HCC)=%.5f HCCcos=%.4f dscore=%.4f"
            % (
                "PASS" if ok else "FAIL",
                file,
                probability_delta,
                cosine_delta,
                map_delta,
                attention_delta,
                float(probs[positive_index]),
                hcc_cosine,
                delta,
            )
        )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-convert", action="store_true", help="reuse the existing .onnx")
    args = parser.parse_args()

    model_path = MODEL_DIR / ONNX_NAME
    predictor = ExplainableHCCPredictor(weights_path=WEIGHTS_PATH, probe_dir=PROBE_DIR)

    if args.skip_convert and model_path.exists():
        print("[onnx] reusing existing artifact (--skip-convert)")
    else:
        with tempfile.TemporaryDirectory() as temporary:
            saved_model_dir = Path(temporary) / "saved_model"
            save_saved_model(predictor, saved_model_dir)
            convert_to_onnx(saved_model_dir, model_path)

    manifest = write_examples()
    write_expected(manifest)
    failures = verify_onnx(model_path)
    # The tfjs attempt left shards behind; the page only needs the ONNX graph.
    for stale in MODEL_DIR.glob("group1-shard*.bin"):
        stale.unlink()
    (MODEL_DIR / "model.json").unlink(missing_ok=True)

    print(f"[onnx] {'ALL PASS' if failures == 0 else f'{failures} FAILED'}")
    raise SystemExit(0 if failures == 0 else 1)


if __name__ == "__main__":
    main()
