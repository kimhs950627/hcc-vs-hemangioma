"""
inference/kaggle_explainable_predictor.py
=========================================
Kaggle-Optimized Explainable Inference Module for HCC vs. Hemangioma Classification.

Designed specifically for Kaggle Notebooks & Scripts:
1. Auto-Discovery:
   - Recursively locates weights (*.weights.h5) and prototype banks (*bank*.npy)
     in `/kaggle/input/` or `/kaggle/working/`.
   - Locates `conv_hybrid_vit.py` if not in python path.
2. Dual-Output Prediction:
   - Confidence P(HCC) vs. Validation Cutoff (0.001088)
   - Cosine Similarity Scores (HCC, Hemangioma, Δscore) vs. Validation Cutoffs (-0.004843, -0.958471)
3. Explainable Visualizations:
   - Grad-CAM heatmap & overlay
   - ViT Attention Rollout across Cross-Attention layers
   - Metric scorecard dashboard with cutoff comparison
4. Kaggle-Friendly Execution:
   - Displays inline plots when run in Kaggle Notebooks (`plt.show()`)
   - Automatically saves reports to `/kaggle/working/`
   - Includes `predict_batch()` to evaluate whole test folders and export CSV.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Union, Dict, Any, Optional, List, Tuple
import importlib.util

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras


# ──────────────────────────────────────────────────────────────────────────────
# 1. Path Auto-Discovery & Module Resolver for Kaggle
# ──────────────────────────────────────────────────────────────────────────────

def get_kaggle_candidate_dirs() -> List[Path]:
    """Return prioritized candidate search roots for Kaggle & local environments."""
    candidates = []
    # Kaggle input & working
    if Path("/kaggle/input").exists():
        candidates.append(Path("/kaggle/input"))
    if Path("/kaggle/working").exists():
        candidates.append(Path("/kaggle/working"))

    # Current script directory and parent tree
    curr = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates.extend([curr, curr.parent, curr.parent.parent])
    candidates.append(Path.cwd())

    # De-duplicate while preserving order
    unique = []
    seen = set()
    for c in candidates:
        res = c.resolve()
        if res not in seen and res.exists():
            seen.add(res)
            unique.append(res)
    return unique


def find_file_auto(
    patterns: Union[str, List[str]],
    search_dirs: Optional[List[Path]] = None
) -> Optional[Path]:
    """Recursively search for a file matching pattern(s) across candidate directories."""
    if isinstance(patterns, str):
        patterns = [patterns]

    if search_dirs is None:
        search_dirs = get_kaggle_candidate_dirs()

    for root in search_dirs:
        for pat in patterns:
            # Direct check
            direct = root / pat
            if direct.exists() and direct.is_file():
                return direct

            # Recursive search
            try:
                for match in root.rglob(pat):
                    if match.is_file():
                        return match
            except Exception:
                continue
    return None


def resolve_conv_hybrid_vit():
    """Ensure `build_conv_hybrid_vit` is accessible, dynamically finding it if necessary."""
    try:
        from models.conv_hybrid_vit import build_conv_hybrid_vit
        return build_conv_hybrid_vit
    except ImportError:
        pass

    vit_file = find_file_auto("conv_hybrid_vit.py")
    if vit_file is not None:
        parent_dir = str(vit_file.parent.parent)
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        try:
            from models.conv_hybrid_vit import build_conv_hybrid_vit
            return build_conv_hybrid_vit
        except Exception:
            spec = importlib.util.spec_from_file_location("conv_hybrid_vit", vit_file)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return getattr(mod, "build_conv_hybrid_vit")

    raise ImportError(
        "Could not find 'conv_hybrid_vit.py'. Please make sure the project directory or dataset "
        "containing conv_hybrid_vit.py is attached to /kaggle/input or sys.path."
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Self-Contained Attention Rollout Helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_attention_rollout_ca(attn_weights_list: list) -> np.ndarray:
    """Compute Cross-Attention rollout over layers [B, H, 1, N] -> [B, N]."""
    per_layer = []
    for a in attn_weights_list:
        a_np = a.numpy() if hasattr(a, "numpy") else np.asarray(a, dtype=np.float32)
        # Head mean over CA shape [B, H, 1, N] -> [B, N]
        layer_map = a_np.mean(axis=1)[:, 0, :]
        row_sum = layer_map.sum(axis=-1, keepdims=True) + 1e-8
        per_layer.append(layer_map / row_sum)
    accumulated = per_layer[0]
    for i, lm in enumerate(per_layer[1:], start=1):
        alpha = i / (i + 1)
        accumulated = alpha * accumulated + (1 - alpha) * lm
    return accumulated


class SupconClassifier(keras.Model):
    """Classifier model holding encoder and classification / projection heads."""
    def __init__(self, encoder, classifier_head, projection_head):
        super().__init__()
        self.encoder = encoder
        self.classifier_head = classifier_head
        self.projection_head = projection_head

    def call(self, x, training=False):
        enc = self.encoder(x, training=training)
        base_embedding = enc["embedding"]
        logits = self.classifier_head(base_embedding, training=training)
        projection = self.projection_head(base_embedding, training=training)
        return {
            "embedding": base_embedding,
            "projection": projection,
            "logits": logits,
            "probabilities": tf.nn.softmax(logits, axis=-1),
            "tokens": enc.get("tokens", None),
            "features": enc.get("features", None),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 3. Kaggle Explainable Predictor Class
# ──────────────────────────────────────────────────────────────────────────────

class KaggleExplainablePredictor:
    """Explainable Inference Engine for HCC vs. Hemangioma Classification on Kaggle."""

    # Youden's J optimal cutoffs from validation set
    DEFAULT_CUTOFFS = {
        "confidence_score": 0.001088,
        "hcc_cosine_score": -0.004843,
        "delta_score": -0.958471,
    }

    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        probe_dir: Optional[Union[str, Path]] = None,
        image_size: int = 384,
        cutoffs: Optional[Dict[str, float]] = None,
        prototype_mode: str = "mean",
    ):
        self.image_size = image_size
        self.cutoffs = cutoffs or self.DEFAULT_CUTOFFS.copy()
        self.prototype_mode = prototype_mode.lower()

        # 1. Locate weights file
        if weights_path is not None and Path(weights_path).exists():
            self.weights_path = Path(weights_path)
        else:
            found_w = find_file_auto([
                "classifier_y0taaeki_BM.weights.h5",
                "*y0taaeki*.weights.h5",
                "*.weights.h5"
            ])
            if found_w is None:
                raise FileNotFoundError(
                    "Weights file (*.weights.h5) not found. Provide `weights_path` or attach dataset to /kaggle/input."
                )
            self.weights_path = found_w

        print(f"[KagglePredictor] Weight file: {self.weights_path}")

        # 2. Locate prototype banks
        if probe_dir is not None and Path(probe_dir).exists():
            probe_path = Path(probe_dir)
        else:
            probe_path = self.weights_path.parent

        hcc_name = f"hcc_bank_{self.prototype_mode}.npy"
        hem_name = f"hem_bank_{self.prototype_mode}.npy"

        hcc_bank_file = probe_path / hcc_name if (probe_path / hcc_name).exists() else find_file_auto(hcc_name)
        hem_bank_file = probe_path / hem_name if (probe_path / hem_name).exists() else find_file_auto(hem_name)

        if hcc_bank_file is None or hem_bank_file is None:
            # Fallback to mean/kmeans
            alt_mode = "kmeans" if self.prototype_mode == "mean" else "mean"
            hcc_bank_file = find_file_auto(f"hcc_bank_{alt_mode}.npy")
            hem_bank_file = find_file_auto(f"hem_bank_{alt_mode}.npy")
            if hcc_bank_file is not None and hem_bank_file is not None:
                print(f"[KagglePredictor] Using fallback prototype mode: {alt_mode}")
                self.prototype_mode = alt_mode

        if hcc_bank_file is None or hem_bank_file is None:
            raise FileNotFoundError(f"Could not find prototype bank files in {probe_path} or search paths.")

        print(f"[KagglePredictor] Prototype banks: {hcc_bank_file.parent}")
        self.hcc_bank = np.load(hcc_bank_file)
        self.hem_bank = np.load(hem_bank_file)

        # L2-normalize prototypes
        if self.hcc_bank.ndim == 1:
            self.hcc_bank = self.hcc_bank[np.newaxis, :]
        if self.hem_bank.ndim == 1:
            self.hem_bank = self.hem_bank[np.newaxis, :]
        self.hcc_bank = self.hcc_bank / (np.linalg.norm(self.hcc_bank, axis=-1, keepdims=True) + 1e-8)
        self.hem_bank = self.hem_bank / (np.linalg.norm(self.hem_bank, axis=-1, keepdims=True) + 1e-8)

        # 3. Build model and load weights
        self._build_and_load_model()

    def _build_and_load_model(self):
        build_fn = resolve_conv_hybrid_vit()

        d_model = 128
        n_patches = (self.image_size // 32) ** 2  # 144

        self.encoder = build_fn(
            input_shape=(self.image_size, self.image_size, 3),
            backbone_name="efficientnetv2_b0",
            d_model=d_model,
            n_patches=n_patches,
            depth=4,
            num_heads=8,
            mlp_dim=384,
            token_attention_mode="ca",
            pool_mode="cls",
            pe_mode="sinusoidal",
        )

        classifier_hidden_dim = 512
        projection_dim = 256
        num_classes = 2

        self.projection_head = keras.Sequential([
            keras.layers.Dense(classifier_hidden_dim, activation="gelu"),
            keras.layers.Dense(projection_dim),
        ], name="stage2_projection_head_supcon")

        self.classifier_head = keras.Sequential([
            keras.layers.Dense(classifier_hidden_dim, activation="gelu"),
            keras.layers.Dense(num_classes),
        ], name="classifier_head_classification")

        self.model = SupconClassifier(
            encoder=self.encoder,
            classifier_head=self.classifier_head,
            projection_head=self.projection_head
        )

        # Build graph
        dummy = tf.zeros((1, self.image_size, self.image_size, 3))
        _ = self.model(dummy, training=False)

        # Load weights
        self.model.load_weights(str(self.weights_path))
        print("[KagglePredictor] Model weights loaded successfully.")

    def preprocess_image(self, image_input: Union[str, Path, np.ndarray, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """Convert arbitrary input into normalized float32 tensor [384, 384, 3] in [0, 255] and display RGB."""
        if isinstance(image_input, (str, Path)):
            p = str(image_input)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Image not found: {p}")
            bgr = cv2.imread(p)
            if bgr is None:
                raise ValueError(f"cv2.imread failed on: {p}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            rgb = image_input.copy()
            if rgb.ndim == 2:
                rgb = np.stack([rgb] * 3, axis=-1)
            elif rgb.ndim == 3 and rgb.shape[2] == 1:
                rgb = np.concatenate([rgb] * 3, axis=-1)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        img_float = resized.astype(np.float32)
        # Keep scale in [0, 255] for consistency with backbone preprocessing
        if img_float.max() <= 1.0:
            img_float = img_float * 255.0

        return img_float, resized

    def compute_gradcam(self, img_tensor_np: np.ndarray, target_class: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Grad-CAM activation map and colored overlay."""
        H, W = img_tensor_np.shape[:2]
        img_display = np.clip(img_tensor_np / 255.0, 0.0, 1.0)
        x_var = tf.Variable(img_tensor_np[None], dtype=tf.float32, trainable=True)

        patch_embed = self.encoder.patch_embed
        norm_layer = self.encoder.norm
        pool_mode = getattr(self.encoder, "pool_mode", "cls")
        classifier_head = self.model.classifier_head

        with tf.GradientTape() as tape:
            tokens, fmap, gh, gw = patch_embed(x_var, training=False)
            B = tf.shape(x_var)[0]
            d = patch_embed.d_model
            N = gh * gw
            patches = tf.reshape(fmap, [B, N, d])

            attn_mode = getattr(patch_embed, "token_attention_mode", "sa")
            if attn_mode == "ca":
                cls_tok = tf.reduce_mean(patches, axis=1, keepdims=True)
            else:
                cls_tok = tf.cast(tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype)

            seq = tf.concat([cls_tok, patches], axis=1)
            if patch_embed.pos_embed is not None:
                seq = seq + tf.cast(patch_embed.pos_embed, seq.dtype)

            for block in self.encoder.blocks:
                res = block(seq, training=False, return_attention=False)
                seq = res if not isinstance(res, tuple) else res[0]

            seq = norm_layer(seq, training=False)
            emb = seq[:, 0, :] if pool_mode == "cls" else tf.reduce_mean(seq[:, 1:], axis=1)
            logits = classifier_head(emb, training=False)
            score = logits[:, target_class]

        grads = tape.gradient(score, fmap)
        if grads is None:
            return np.zeros((H, W), dtype=np.float32), img_display

        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * fmap, axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam_resized = tf.image.resize(cam[..., None], [H, W]).numpy()[0, :, :, 0].astype(np.float32)

        heatmap_rgb = plt.get_cmap("jet")(cam_resized)[:, :, :3]
        overlay = np.clip(0.55 * img_display + 0.45 * heatmap_rgb, 0.0, 1.0).astype(np.float32)
        return cam_resized, overlay

    def compute_attention_map(self, img_tensor_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Cross-Attention weights and compute spatial rollout map."""
        H, W = img_tensor_np.shape[:2]
        img_display = np.clip(img_tensor_np / 255.0, 0.0, 1.0)
        x = tf.constant(img_tensor_np[None], dtype=tf.float32)

        enc_out = self.encoder(x, training=False, return_attention=True)
        attn_weights = enc_out.get("attention_weights", None)

        if attn_weights is None or len(attn_weights) == 0:
            return np.zeros((H, W), dtype=np.float32), img_display

        rollout_vec = compute_attention_rollout_ca(attn_weights)[0]
        side = int(round(np.sqrt(len(rollout_vec))))
        rollout_map = rollout_vec.reshape((side, side))

        rollout_map = rollout_map - rollout_map.min()
        rollout_map = rollout_map / (rollout_map.max() + 1e-8)

        rollout_resized = cv2.resize(rollout_map, (W, H), interpolation=cv2.INTER_CUBIC)
        rollout_resized = np.clip(rollout_resized, 0.0, 1.0)

        heatmap_rgb = plt.get_cmap("magma")(rollout_resized)[:, :, :3]
        overlay = np.clip(0.55 * img_display + 0.45 * heatmap_rgb, 0.0, 1.0).astype(np.float32)
        return rollout_resized, overlay

    def predict_and_explain(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        save_plot_path: Optional[Union[str, Path]] = None,
        show_inline: bool = True,
        title_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run full explainable prediction for a single ultrasound image."""
        img_tensor_np, img_display = self.preprocess_image(image_input)
        x = tf.constant(img_tensor_np[None], dtype=tf.float32)

        # 1. Forward Pass
        out = self.model(x, training=False)
        probs = out["probabilities"].numpy()[0]
        emb = out["embedding"].numpy()[0]

        p_hem = float(probs[0])
        p_hcc = float(probs[1])

        # 2. Embedding Cosine Similarity
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        hcc_sims = np.dot(self.hcc_bank, emb_norm)
        hem_sims = np.dot(self.hem_bank, emb_norm)

        hcc_cos = float(np.mean(hcc_sims))
        hem_cos = float(np.mean(hem_sims))
        delta_score = float(hcc_cos - hem_cos)

        # 3. Cutoff Comparisons
        pred_conf = "HCC" if p_hcc >= self.cutoffs["confidence_score"] else "Hemangioma"
        pred_cos = "HCC" if hcc_cos >= self.cutoffs["hcc_cosine_score"] else "Hemangioma"
        pred_delta = "HCC" if delta_score >= self.cutoffs["delta_score"] else "Hemangioma"

        # 4. Explainability maps
        cam_map, cam_overlay = self.compute_gradcam(img_tensor_np, target_class=1)
        attn_map, attn_overlay = self.compute_attention_map(img_tensor_np)

        # 5. Format Title & Summary
        title_str = title_prefix or (Path(image_input).name if isinstance(image_input, (str, Path)) else "Ultrasound Examination")

        summary_text = (
            f"==========================================================\n"
            f"  Case: {title_str}\n"
            f"==========================================================\n"
            f"1. Confidence Score (Softmax Output):\n"
            f"   - P(HCC)            : {p_hcc:.6f}  (Cutoff: {self.cutoffs['confidence_score']:.6f})\n"
            f"   - P(Hemangioma)     : {p_hem:.6f}\n"
            f"   - Decision          : [{pred_conf}]\n\n"
            f"2. Embedding Similarity Scores ({self.prototype_mode.upper()} Prototype):\n"
            f"   - HCC Cosine Score  : {hcc_cos:.4f}  (Cutoff: {self.cutoffs['hcc_cosine_score']:.4f}) -> [{pred_cos}]\n"
            f"   - Hemangioma Score  : {hem_cos:.4f}\n"
            f"   - Δscore (Margin)   : {delta_score:.4f}  (Cutoff: {self.cutoffs['delta_score']:.4f}) -> [{pred_delta}]\n\n"
            f"3. Consensus Decision  : {pred_conf} ({'Malignant' if pred_conf == 'HCC' else 'Benign'})\n"
            f"=========================================================="
        )

        results = {
            "title": title_str,
            "confidence_score": p_hcc,
            "p_hemangioma": p_hem,
            "hcc_cosine_score": hcc_cos,
            "hem_cosine_score": hem_cos,
            "delta_score": delta_score,
            "predictions": {
                "by_confidence": pred_conf,
                "by_hcc_cosine": pred_cos,
                "by_delta": pred_delta,
            },
            "cutoffs": self.cutoffs,
            "summary": summary_text,
            "gradcam_overlay": cam_overlay,
            "attention_overlay": attn_overlay,
            "original_image": img_display,
        }

        # 6. Save & Plot Report
        default_save_path = save_plot_path
        if default_save_path is None and Path("/kaggle/working").exists():
            default_save_path = Path("/kaggle/working") / f"report_{Path(title_str).stem}.png"

        self._plot_dashboard(results, save_path=default_save_path, show_inline=show_inline)

        return results

    def _plot_dashboard(
        self,
        res: Dict[str, Any],
        save_path: Optional[Union[str, Path]] = None,
        show_inline: bool = True
    ):
        """Render high-resolution 4-panel diagnostic dashboard."""
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=150)
        title = res["title"]

        # Panel 1: Original Image
        axes[0].imshow(res["original_image"])
        axes[0].set_title(f"Original Ultrasound ({self.image_size}x{self.image_size})", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM Overlay
        axes[1].imshow(res["gradcam_overlay"])
        axes[1].set_title("Grad-CAM Activation (HCC class)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Attention Weight Rollout Overlay
        axes[2].imshow(res["attention_overlay"])
        axes[2].set_title("ViT Attention Weight Rollout", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        # Panel 4: Metrics Dashboard Bar Plot
        ax_dash = axes[3]
        metrics = ["Confidence\nP(HCC)", "HCC Cosine\nScore", "Hemangioma\nCosine Score", "Δscore\n(Margin)"]
        values = [res["confidence_score"], res["hcc_cosine_score"], res["hem_cosine_score"], res["delta_score"]]
        cutoffs = [res["cutoffs"]["confidence_score"], res["cutoffs"]["hcc_cosine_score"], np.nan, res["cutoffs"]["delta_score"]]
        colors = ["#d9534f" if v >= c else "#5bc0de" for v, c in zip(values, cutoffs)]

        bars = ax_dash.bar(range(len(metrics)), values, color=colors, width=0.55, edgecolor="#222222", linewidth=1.2)
        ax_dash.set_xticks(range(len(metrics)))
        ax_dash.set_xticklabels(metrics, fontsize=9.5)
        ax_dash.set_ylabel("Metric Value", fontsize=10, fontweight="bold")
        ax_dash.set_title("Dual-Output Scorecard", fontsize=11, fontweight="bold")
        ax_dash.axhline(0, color="gray", linewidth=0.8, linestyle="--")

        # Value annotations
        for idx, (bar, val, cut) in enumerate(zip(bars, values, cutoffs)):
            y_pos = bar.get_height()
            offset = 0.05 if y_pos >= 0 else -0.15
            txt = f"{val:.3f}"
            if not np.isnan(cut):
                txt += f"\n(cut: {cut:.3f})"
            ax_dash.text(bar.get_x() + bar.get_width() / 2.0, y_pos + offset, txt, ha="center", va="bottom" if offset > 0 else "top", fontsize=8.5, fontweight="bold")

        ax_dash.set_ylim(min(min(values) - 0.3, -1.2), max(max(values) + 0.3, 1.2))
        ax_dash.grid(axis="y", linestyle=":", alpha=0.5)

        plt.suptitle(f"{title} | Final Decision: {res['predictions']['by_confidence']}", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path is not None:
            save_p = Path(save_path)
            save_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(str(save_p), bbox_inches="tight")
            print(f"[KagglePredictor] Diagnostic report saved to: {save_p}")

        if show_inline:
            plt.show()
        else:
            plt.close()

    def predict_batch(
        self,
        image_dir_or_files: Union[str, Path, List[Union[str, Path]]],
        output_csv: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        save_reports_dir: Optional[Union[str, Path]] = None,
    ):
        """Batch evaluate images from a folder or list and export a summary CSV."""
        import pandas as pd

        if isinstance(image_dir_or_files, (str, Path)):
            p = Path(image_dir_or_files)
            if p.is_dir():
                valid_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
                files_to_eval = []
                for ext in valid_exts:
                    files_to_eval.extend(list(p.rglob(ext)))
                files_to_eval = sorted(files_to_eval)
            else:
                files_to_eval = [p]
        else:
            files_to_eval = list(image_dir_or_files)

        if max_samples is not None:
            files_to_eval = files_to_eval[:max_samples]

        print(f"[KagglePredictor] Running batch evaluation on {len(files_to_eval)} images...")
        rows = []
        for idx, img_p in enumerate(files_to_eval):
            img_path = Path(img_p)
            save_report = None
            if save_reports_dir is not None:
                save_report = Path(save_reports_dir) / f"{img_path.stem}_report.png"

            try:
                res = self.predict_and_explain(
                    image_input=img_path,
                    save_plot_path=save_report,
                    show_inline=False,
                    title_prefix=img_path.name
                )
                rows.append({
                    "image_path": str(img_path),
                    "filename": img_path.name,
                    "P_HCC": res["confidence_score"],
                    "P_Hemangioma": res["p_hemangioma"],
                    "HCC_Cosine": res["hcc_cosine_score"],
                    "Hem_Cosine": res["hem_cosine_score"],
                    "Delta_Score": res["delta_score"],
                    "Pred_Confidence": res["predictions"]["by_confidence"],
                    "Pred_HCC_Cosine": res["predictions"]["by_hcc_cosine"],
                    "Pred_Delta": res["predictions"]["by_delta"],
                })
            except Exception as e:
                print(f"Error evaluating {img_path}: {e}")

        df = pd.DataFrame(rows)
        if output_csv is not None:
            out_p = Path(output_csv)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out_p, index=False)
            print(f"[KagglePredictor] Batch summary exported to: {out_p}")

        return df


# ──────────────────────────────────────────────────────────────────────────────
# 4. CLI Execution
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kaggle Explainable Inference for HCC vs. Hemangioma")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to ultrasound image file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save explanation report")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Path to weights file")
    parser.add_argument("--batch_dir", "-b", type=str, default=None, help="Directory to run batch evaluation on")
    parser.add_argument("--title", "-t", type=str, default="US Examination Case", help="Report title")
    args = parser.parse_args()

    predictor = KaggleExplainablePredictor(weights_path=args.weights)

    if args.batch_dir is not None:
        csv_out = args.output or (Path("/kaggle/working/predictions.csv") if Path("/kaggle/working").exists() else "batch_predictions.csv")
        predictor.predict_batch(args.batch_dir, output_csv=csv_out)
    else:
        test_img = args.image
        if test_img is None:
            sample_candidate = find_file_auto("hcc1.jpg")
            if sample_candidate is not None:
                test_img = str(sample_candidate)
            else:
                print("No image provided. Specify `--image <path>`.")
                sys.exit(0)

        out_p = args.output
        res = predictor.predict_and_explain(test_img, save_plot_path=out_p, show_inline=False, title_prefix=args.title)
        print(res["summary"])
