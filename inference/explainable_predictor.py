"""
inference/explainable_predictor.py
==================================
Explainable Prediction Module for HCC vs. Hemangioma Ultrasound Classification.

Loads the primary model (EfficientNetV2B0 + CE + SupCon) and produces:
1. Confidence score (softmax probability P(HCC)) with cutoff comparison
2. Embedding-based Cosine Scores (HCC Cosine Score, Hemangioma Cosine Score, Δscore) with cutoffs
3. Grad-CAM visual explanation overlay
4. ViT Attention weight / rollout map overlay
5. Unified visual report generation

Usage:
------
    from inference.explainable_predictor import ExplainableHCCPredictor

    predictor = ExplainableHCCPredictor()
    results = predictor.predict_and_explain(
        image_path="example_images_hcc_case/hcc1.jpg",
        save_plot_path="hcc1_explanation.png"
    )
    print(results["summary"])
"""

import os
import sys
from pathlib import Path
from typing import Union, Dict, Any, Optional

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import keras

# Root path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.conv_hybrid_vit import build_conv_hybrid_vit

def _compute_attention_rollout_ca(attn_weights_list: list) -> np.ndarray:
    """Self-contained CA attention rollout over layers [B, H, 1, N] -> [B, N]."""
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
    return accumulated # [B, N]


class SupconClassifier(keras.Model):
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


class ExplainableHCCPredictor:
    """End-to-end Explainable Predictor for HCC vs Hemangioma B-mode Ultrasound."""

    # Default operating cutoffs determined by Youden's J on the validation set
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
    ):
        self.image_size = image_size
        self.cutoffs = cutoffs or self.DEFAULT_CUTOFFS.copy()

        # Paths
        if weights_path is None:
            weights_path = (
                PROJECT_ROOT
                / "cosine_probe_result"
                / "effnet_supcon"
                / "classifier_y0taaeki_BM.weights.h5"
            )
        self.weights_path = Path(weights_path)

        if probe_dir is None:
            probe_dir = PROJECT_ROOT / "cosine_probe_result" / "effnet_supcon"
        self.probe_dir = Path(probe_dir)

        # 1. Load Prototypes
        self._load_prototypes()

        # 2. Build and Load Model
        self._build_and_load_model()

    def _load_prototypes(self):
        """Load training set HCC and Hemangioma prototypes."""
        hcc_proto_path = self.probe_dir / "hcc_bank_mean.npy"
        hem_proto_path = self.probe_dir / "hem_bank_mean.npy"

        if not hcc_proto_path.exists() or not hem_proto_path.exists():
            raise FileNotFoundError(
                f"Prototype files not found in {self.probe_dir}. "
                "Expected hcc_bank_mean.npy and hem_bank_mean.npy."
            )

        self.hcc_prototype = np.load(hcc_proto_path)[0]  # shape (128,)
        self.hem_prototype = np.load(hem_proto_path)[0]  # shape (128,)

        # Normalize prototypes
        self.hcc_proto_norm = self.hcc_prototype / (
            np.linalg.norm(self.hcc_prototype) + 1e-8
        )
        self.hem_proto_norm = self.hem_prototype / (
            np.linalg.norm(self.hem_prototype) + 1e-8
        )

    def _build_and_load_model(self):
        """Construct ConvHybridViT architecture and load pre-trained weights."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")

        print(f"[Predictor] Building ConvHybridViT ({self.image_size}x{self.image_size})...")
        self.encoder = build_conv_hybrid_vit(
            input_shape=(self.image_size, self.image_size, 3),
            backbone_name="efficientnetv2_b0",
            d_model=128,
            n_patches=144,
            depth=4,
            num_heads=8,
            mlp_dim=384,
            dropout=0.1,
            imagenet_pretrained=False,
            backbone_trainable=True,
            use_sinusoidal_pe=True,
            pool_mode="cls",
            pe_mode="sinusoidal",
            token_attention_mode="ca",
        )

        projection_head = keras.Sequential(
            [
                keras.layers.Dense(512, activation="gelu"),
                keras.layers.Dense(256),
            ],
            name="stage2_projection_head_supcon",
        )

        classifier_head = keras.Sequential(
            [
                keras.layers.Dense(512, activation="gelu"),
                keras.layers.Dense(2),
            ],
            name="classifier_head_classification",
        )

        self.model = SupconClassifier(
            encoder=self.encoder,
            classifier_head=classifier_head,
            projection_head=projection_head,
        )

        # Warm start
        _dummy = tf.zeros((2, self.image_size, self.image_size, 3), dtype=tf.float32)
        _ = self.model(_dummy, training=False)

        print(f"[Predictor] Loading weights from: {self.weights_path.name}...")
        self.model.load_weights(str(self.weights_path))
        print("[Predictor] Model loaded successfully.")

    def preprocess_image(
        self, image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load and preprocess image.

        Returns:
            img_tensor_np: (384, 384, 3) float32 in [0, 255] for model
            img_display: (384, 384, 3) float32 in [0, 1] for visualization
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            # Read image (handle potential grayscale or RGB)
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Could not load image via cv2: {img_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            rgb = image_input.copy()
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
            elif rgb.shape[2] == 4:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Resize to input resolution
        resized = cv2.resize(
            rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
        )

        img_tensor_np = resized.astype(np.float32)  # [0, 255]
        img_display = resized.astype(np.float32) / 255.0  # [0, 1]
        return img_tensor_np, img_display

    def compute_gradcam(
        self, img_tensor_np: np.ndarray, target_class: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Grad-CAM activation map and overlay for target class.

        Args:
            img_tensor_np: (384, 384, 3) float32
            target_class: 0 for Hemangioma, 1 for HCC
        Returns:
            cam_norm: (H, W) float32 in [0, 1]
            overlay: (H, W, 3) float32 in [0, 1]
        """
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
                cls_tok = tf.cast(
                    tf.tile(patch_embed.cls_token, [B, 1, 1]), patches.dtype
                )

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
            # Fallback
            return np.zeros((H, W), dtype=np.float32), img_display

        # Channel pooling
        weights = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
        cam = tf.reduce_sum(weights * fmap, axis=-1)
        cam = tf.nn.relu(cam)
        cam = cam / (tf.reduce_max(cam) + 1e-8)
        cam_resized = (
            tf.image.resize(cam[..., None], [H, W]).numpy()[0, :, :, 0].astype(np.float32)
        )

        # Colormap overlay (JET)
        heatmap_rgb = plt.get_cmap("jet")(cam_resized)[:, :, :3]
        overlay = np.clip(0.55 * img_display + 0.45 * heatmap_rgb, 0.0, 1.0).astype(
            np.float32
        )
        return cam_resized, overlay

    def compute_attention_map(
        self, img_tensor_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract ViT cross-attention weights and compute spatial rollout map."""
        H, W = img_tensor_np.shape[:2]
        img_display = np.clip(img_tensor_np / 255.0, 0.0, 1.0)
        x = tf.constant(img_tensor_np[None], dtype=tf.float32)

        enc_out = self.encoder(x, training=False, return_attention=True)
        attn_weights = enc_out.get("attention_weights", None)

        if attn_weights is None or len(attn_weights) == 0:
            return np.zeros((H, W), dtype=np.float32), img_display

        # Compute rollout map using internal helper
        rollout_vec = _compute_attention_rollout_ca(attn_weights)[0]  # (144,)
        side = int(round(np.sqrt(len(rollout_vec))))
        rollout_map = rollout_vec.reshape((side, side))

        # Normalize
        rollout_map = rollout_map - rollout_map.min()
        rollout_map = rollout_map / (rollout_map.max() + 1e-8)

        # Resize to image resolution
        rollout_resized = cv2.resize(rollout_map, (W, H), interpolation=cv2.INTER_CUBIC)
        rollout_resized = np.clip(rollout_resized, 0.0, 1.0)

        # Colormap overlay (magma / inferno)
        heatmap_rgb = plt.get_cmap("magma")(rollout_resized)[:, :, :3]
        overlay = np.clip(0.55 * img_display + 0.45 * heatmap_rgb, 0.0, 1.0).astype(
            np.float32
        )
        return rollout_resized, overlay

    def predict_and_explain(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        save_plot_path: Optional[Union[str, Path]] = None,
        title_prefix: str = "B-Mode Ultrasound Case Analysis",
    ) -> Dict[str, Any]:
        """Run full dual-output inference and explainability pipeline.

        Returns dictionary containing:
        - confidence_score: Softmax P(HCC)
        - p_hemangioma: Softmax P(Hemangioma)
        - hcc_cosine_score: Cosine similarity to mean HCC prototype
        - hem_cosine_score: Cosine similarity to mean Hemangioma prototype
        - delta_score: HCC cosine - Hemangioma cosine
        - predictions: Class determination by each metric against validation cutoffs
        - gradcam_overlay: (H, W, 3) float32 [0, 1]
        - attention_overlay: (H, W, 3) float32 [0, 1]
        - summary: Human-readable text report
        """
        img_tensor_np, img_display = self.preprocess_image(image_input)
        x = tf.constant(img_tensor_np[None], dtype=tf.float32)

        # 1. Forward Pass
        out = self.model(x, training=False)
        logits = out["logits"].numpy()[0]
        probs = out["probabilities"].numpy()[0]
        emb = out["embedding"].numpy()[0]

        p_hem = float(probs[0])
        p_hcc = float(probs[1])

        # 2. Embedding Cosine Similarity
        emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
        hcc_cos = float(np.dot(emb_norm, self.hcc_proto_norm))
        hem_cos = float(np.dot(emb_norm, self.hem_proto_norm))
        delta_score = hcc_cos - hem_cos

        # 3. Cutoff Comparisons
        pred_conf = "HCC" if p_hcc >= self.cutoffs["confidence_score"] else "Hemangioma"
        pred_cos = "HCC" if hcc_cos >= self.cutoffs["hcc_cosine_score"] else "Hemangioma"
        pred_delta = "HCC" if delta_score >= self.cutoffs["delta_score"] else "Hemangioma"

        # 4. Explainability Maps
        cam_map, cam_overlay = self.compute_gradcam(img_tensor_np, target_class=1)
        att_map, att_overlay = self.compute_attention_map(img_tensor_np)

        # 5. Format Summary
        summary = (
            f"==========================================================\n"
            f"  {title_prefix}\n"
            f"==========================================================\n"
            f"1. Confidence Score (Softmax Output):\n"
            f"   - P(HCC)            : {p_hcc:.6f}  (Cutoff: {self.cutoffs['confidence_score']:.6f})\n"
            f"   - P(Hemangioma)     : {p_hem:.6f}\n"
            f"   - Decision          : [{pred_conf}]\n\n"
            f"2. Embedding-based Similarity Scores (Mean Prototype):\n"
            f"   - HCC Cosine Score  : {hcc_cos:.4f}  (Cutoff: {self.cutoffs['hcc_cosine_score']:.4f}) -> [{pred_cos}]\n"
            f"   - Hemangioma Score  : {hem_cos:.4f}\n"
            f"   - Δscore (Margin)   : {delta_score:.4f}  (Cutoff: {self.cutoffs['delta_score']:.4f}) -> [{pred_delta}]\n\n"
            f"3. Consensus Decision  : "
            f"{'HCC (Malignant)' if pred_conf == 'HCC' else 'Hemangioma (Benign)'}\n"
            f"=========================================================="
        )

        res = {
            "confidence_score": p_hcc,
            "p_hemangioma": p_hem,
            "hcc_cosine_score": hcc_cos,
            "hem_cosine_score": hem_cos,
            "delta_score": delta_score,
            "cutoffs": self.cutoffs.copy(),
            "predictions": {
                "by_confidence": pred_conf,
                "by_hcc_cosine": pred_cos,
                "by_delta_score": pred_delta,
            },
            "gradcam_heatmap": cam_map,
            "gradcam_overlay": cam_overlay,
            "attention_map": att_map,
            "attention_overlay": att_overlay,
            "original_image": img_display,
            "summary": summary,
        }

        # 6. Save Plot if requested
        if save_plot_path is not None:
            self._save_diagnostic_report(res, save_plot_path, title_prefix)

        return res

    def _save_diagnostic_report(
        self, res: Dict[str, Any], save_path: Union[str, Path], title: str
    ):
        """Generate a 4-panel diagnostic visual report figure."""
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), dpi=300)

        # Panel 1: Original Image
        axes[0].imshow(res["original_image"])
        axes[0].set_title("Original Ultrasound (384x384)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # Panel 2: Grad-CAM Overlay
        axes[1].imshow(res["gradcam_overlay"])
        axes[1].set_title("Grad-CAM Activation (HCC class)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        # Panel 3: Attention Map Overlay
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

        # Annotate values and cutoffs
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

        save_p = Path(save_path)
        save_p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_p), bbox_inches="tight")
        plt.close()
        print(f"[Predictor] Diagnostic report saved to: {save_p}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Explainable Inference for HCC vs. Hemangioma Ultrasound Classifier")
    parser.add_argument("--image", "-i", type=str, default=str(PROJECT_ROOT / "example_images_hcc_case" / "hcc1.jpg"), help="Path to ultrasound image file")
    parser.add_argument("--output", "-o", type=str, default=str(PROJECT_ROOT / "scratch" / "explainable_prediction_result.png"), help="Path to save explanation report")
    parser.add_argument("--title", "-t", type=str, default="US Examination Case", help="Title for the visual report")
    args = parser.parse_args()

    predictor = ExplainableHCCPredictor()
    test_img = Path(args.image)
    if test_img.exists():
        res = predictor.predict_and_explain(
            test_img,
            save_plot_path=Path(args.output),
            title_prefix=args.title
        )
        print(res["summary"])
    else:
        print(f"Error: Target image file not found: {test_img}")

