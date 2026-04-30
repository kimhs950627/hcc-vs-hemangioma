"""Supervised benchmark baseline for HCC vs. Hemangioma classification.

NO pretraining, NO regularizer (no dropout / weight-decay / label-smoothing
unless explicitly enabled via flags).
Goal: establish a clean scratch-supervised lower-bound for SSL comparison.

Usage
-----
CLI quick-start::

    python training/benchmark_supervised.py \
        --data_root ./clean_ver_for_train \
        --encoder   vit \
        --epochs    100 \
        --batch_size 32 \
        --output_dir outputs/benchmark_vit

Python API::

    from training.benchmark_supervised import run_supervised_benchmark

    run_supervised_benchmark(
        data_root   = "./clean_ver_for_train",
        encoder_name= "vit",          # "vit" | "swin" | "convnext" | "efficientnet"
        input_shape = (224, 224, 3),
        batch_size  = 32,
        epochs      = 100,
        lr          = 1e-4,
        use_augmentation   = False,   # True: base augmentation / False: rescale only
        use_pretrained     = False,   # WARNING: set True only for ablation
        use_dropout        = False,
        use_weight_decay   = False,
        output_dir         = "outputs/benchmark_vit",
        seed               = 42,
    )

Outputs (all under ``output_dir``)::

    checkpoints/          — best weights (.weights.h5)
    metrics/              — per-epoch JSON and final CSV
    roc/                  — ROC curve PNGs per epoch
    gradcam/              — Grad-CAM overlay PNGs per epoch
    attention/            — ViT/Swin attention map PNGs per epoch
    test_report.json      — final test-set metrics
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
from typing import Any

import numpy as np
import tensorflow as tf
import keras
from keras import layers

# ── local imports ─────────────────────────────────────────────────────────────
try:
    from dataloader import build_dataset
except ImportError:
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from dataloader import build_dataset

from models.encoder import build_encoder
from utils.metrics import binary_classification_metrics, save_metrics_json, plot_roc_curve
from callbacks.epoch_visualization import EpochMetricsAndVisualizationCallback
from visualization.wandb_viz import WandbBenchmarkVisualizer, WandbVisualizationConfig


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────

def build_supervised_benchmark_model(
    encoder_name: str = "vit",
    input_shape: tuple[int, int, int] = (224, 224, 3),
    use_pretrained: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.3,
    use_weight_decay: bool = False,
    weight_decay: float = 1e-4,
) -> keras.Model:
    """Build scratch supervised binary classifier.

    Architecture
    ------------
    encoder -> GlobalAveragePooling (CNN) / CLS token (ViT/Swin)
    -> [optional Dropout]
    -> Dense(1, activation=None)   (binary logit)

    By default:
    - No dropout, no weight decay, no label smoothing.
    - CNN encoders (ConvNeXt, EfficientNet): pretrained weights are loaded from
      ImageNet if ``use_pretrained=True`` (off by default for fair benchmark).
    - ViT / Swin: always initialized from scratch (no ImageNet ViT weights
      available in keras.applications).

    Args:
        encoder_name    : One of ``vit``, ``swin``, ``convnext``, ``efficientnet``.
        input_shape     : (H, W, C).
        use_pretrained  : If True, CNN backbones use ImageNet weights.
        use_dropout     : Add dropout before classifier head.
        dropout_rate    : Dropout probability (only if use_dropout=True).
        use_weight_decay: Apply L2 weight decay to Dense head kernel.
        weight_decay    : L2 coefficient (only if use_weight_decay=True).

    Returns:
        A ``keras.Model`` with two outputs signature via ``call()`` returning
        dict with 'logits' (scalar) and encoder outputs.
    """
    # ── Encoder ───────────────────────────────────────────────────────────────
    if use_pretrained:
        encoder = build_encoder(encoder_name, input_shape=input_shape)
    else:
        # For CNN backbones, override weights to None inside build_encoder
        # by monkey-patching keras.applications at call time.
        import keras.applications as _apps
        _orig_cnxt = _apps.ConvNeXtTiny
        _orig_effn = _apps.EfficientNetB0

        def _cnxt_no_pt(**kw):
            kw['weights'] = None
            return _orig_cnxt(**kw)

        def _effn_no_pt(**kw):
            kw['weights'] = None
            return _orig_effn(**kw)

        _apps.ConvNeXtTiny  = _cnxt_no_pt   # type: ignore[attr-defined]
        _apps.EfficientNetB0 = _effn_no_pt  # type: ignore[attr-defined]
        try:
            encoder = build_encoder(encoder_name, input_shape=input_shape)
        finally:
            _apps.ConvNeXtTiny  = _orig_cnxt
            _apps.EfficientNetB0 = _orig_effn

    # ── Head ──────────────────────────────────────────────────────────────────
    kernel_regularizer = (
        keras.regularizers.L2(weight_decay) if use_weight_decay else None
    )

    class _BenchmarkClassifier(keras.Model):
        """Thin wrapper: encoder + optional dropout + single logit head."""

        def __init__(self, enc: keras.Model, *, drop: bool, rate: float, kreg: Any):
            super().__init__(name='benchmark_classifier')
            self.enc     = enc
            self.dropout = layers.Dropout(rate) if drop else None
            self.head    = layers.Dense(
                1, activation=None,
                kernel_regularizer=kreg,
                name='binary_logit',
            )

        def call(self, x: tf.Tensor, training: bool = False) -> dict[str, Any]:
            enc_out  = self.enc(x, training=training)
            embedding = enc_out['embedding']           # (B, D)
            if self.dropout is not None:
                embedding = self.dropout(embedding, training=training)
            logit = self.head(embedding)               # (B, 1)
            prob  = tf.sigmoid(logit)                  # (B, 1)
            return {
                'logit'      : logit,
                'probability': prob,
                'embedding'  : enc_out['embedding'],
                # Pass through encoder-specific keys for visualization
                'feature_map': enc_out.get('feature_map'),
                'last_encoder_layer_attentional_weights':
                    enc_out.get('last_encoder_layer_attentional_weights'),
            }

    return _BenchmarkClassifier(
        encoder,
        drop=use_dropout,
        rate=dropout_rate,
        kreg=kernel_regularizer,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Keras train_step wrapper
# ─────────────────────────────────────────────────────────────────────────────

class _SupervisedTrainer(keras.Model):
    """Wraps the benchmark model for .fit() compatibility.

    Inputs expected by .fit()::

        (images: float32 [B, H, W, 3],  labels: int32 [B])   -> label in {0, 1}

    Compiles with::

        trainer.compile(
            optimizer = ...,
            loss      = ...,
            metrics   = ...,
        )
    """

    def __init__(self, core: keras.Model):
        super().__init__(name='supervised_trainer')
        self.core = core

    def call(self, x: tf.Tensor, training: bool = False) -> dict[str, Any]:
        return self.core(x, training=training)

    def predict_proba(self, dataset: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
        """Run inference on a tf.data.Dataset, return (y_true, y_prob)."""
        all_true, all_prob = [], []
        for imgs, lbls in dataset:
            out   = self.core(imgs, training=False)
            probs = tf.squeeze(out['probability'], axis=-1).numpy()  # (B,)
            all_true.append(lbls.numpy())
            all_prob.append(probs)
        return np.concatenate(all_true), np.concatenate(all_prob)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_supervised_benchmark(
    data_root: str,
    encoder_name: str = "vit",
    input_shape: tuple[int, int, int] = (224, 224, 3),
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-4,
    use_augmentation: bool = False,
    use_pretrained: bool = False,
    use_dropout: bool = False,
    dropout_rate: float = 0.3,
    use_weight_decay: bool = False,
    weight_decay: float = 1e-4,
    output_dir: str = "outputs/benchmark",
    seed: int = 42,
    viz_every_n_epochs: int = 1,
    max_viz_samples: int = 8,
) -> dict[str, Any]:
    """Run scratch supervised benchmark and return test metrics.

    Args:
        data_root          : Path to ``clean_ver_for_train/`` directory.
        encoder_name       : ``vit`` | ``swin`` | ``convnext`` | ``efficientnet``.
        input_shape        : (H, W, C) — must match encoder expectation.
        batch_size         : Mini-batch size.
        epochs             : Total training epochs.
        lr                 : Adam learning rate.
        use_augmentation   : Apply base augmentation on train split (default False
                             for a pure benchmark; True for augmented ablation).
        use_pretrained     : CNN backbone ImageNet pretraining flag.
        use_dropout        : Add Dropout(dropout_rate) before head.
        dropout_rate       : Dropout probability.
        use_weight_decay   : L2 regularization on head kernel.
        weight_decay       : L2 coefficient.
        output_dir         : Root directory for all saved artifacts.
        seed               : Random seed.
        viz_every_n_epochs : Run Grad-CAM / attention callback every N epochs.
        max_viz_samples    : Max images to visualize per epoch.

    Returns:
        dict with final test-set metrics.
    """
    # ── Reproducibility ───────────────────────────────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    outdir = pathlib.Path(output_dir)
    for sub in ('checkpoints', 'metrics', 'roc', 'reports', 'attention'):
        (outdir / sub).mkdir(parents=True, exist_ok=True)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print(f"[benchmark] Building datasets from: {data_root}")
    ds_train, ds_val, ds_test = build_dataset(
        data_root        = data_root,
        img_size         = input_shape[:2],
        batch_size       = batch_size,
        use_augmentation = use_augmentation,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"[benchmark] Building encoder: {encoder_name}  pretrained={use_pretrained}")
    core_model = build_supervised_benchmark_model(
        encoder_name   = encoder_name,
        input_shape    = input_shape,
        use_pretrained = use_pretrained,
        use_dropout    = use_dropout,
        dropout_rate   = dropout_rate,
        use_weight_decay = use_weight_decay,
        weight_decay   = weight_decay,
    )
    trainer = _SupervisedTrainer(core_model)

    # ── Compile ───────────────────────────────────────────────────────────────
    # Wrap model output so that .fit() receives standard (y_pred, y_true) signature.
    # We build a thin functional model around the trainer for compile/fit.
    inp       = keras.Input(shape=input_shape, name='image')
    out_dict  = core_model(inp, training=False)
    logit_out = out_dict['logit']                         # (B, 1)

    fit_model = keras.Model(inputs=inp, outputs=logit_out, name='benchmark_fit_model')
    fit_model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr),
        loss      = keras.losses.BinaryCrossentropy(from_logits=True),
        metrics   = [
            keras.metrics.BinaryAccuracy(name='accuracy', threshold=0.0),
        ],
    )

    # ── Visualization callback ─────────────────────────────────────────────────
    # Grab a fixed sample batch for visualization.
    sample_images, sample_labels = next(iter(ds_val))
    sample_images = sample_images[:max_viz_samples]
    sample_labels = sample_labels[:max_viz_samples]

    # Determine last conv layer name for Grad-CAM (CNN only).
    last_conv = _infer_last_conv_layer(core_model)

    viz_callback = EpochMetricsAndVisualizationCallback(
        val_dataset          = ds_val,
        output_dir           = str(outdir),
        last_conv_layer_name = last_conv,
        core_model           = core_model,
        sample_images        = sample_images.numpy(),
        sample_labels        = sample_labels.numpy(),
        viz_every_n_epochs   = viz_every_n_epochs,
        max_viz_samples      = max_viz_samples,
    )

    # Best checkpoint callback.
    ckpt_path = str(outdir / 'checkpoints' / 'best.weights.h5')
    ckpt_callback = keras.callbacks.ModelCheckpoint(
        filepath         = ckpt_path,
        monitor          = 'val_accuracy',
        save_best_only   = True,
        save_weights_only= True,
        verbose          = 1,
    )

    # ── Training ──────────────────────────────────────────────────────────────
    print("[benchmark] Starting training ...")
    fit_model.fit(
        ds_train,
        validation_data = ds_val,
        epochs          = epochs,
        callbacks       = [viz_callback, ckpt_callback],
        verbose         = 1,
    )

    # ── Final test evaluation ──────────────────────────────────────────────────
    print("[benchmark] Loading best checkpoint for test evaluation ...")
    fit_model.load_weights(ckpt_path)

    # Collect test predictions from the core_model.
    y_true_test, y_prob_test = trainer.predict_proba(ds_test)
    test_metrics = binary_classification_metrics(y_true_test, y_prob_test)
    test_report  = outdir / 'test_report.json'
    save_metrics_json(test_metrics, test_report)
    print(f"[benchmark] Test metrics saved → {test_report}")
    _print_metrics(test_metrics, prefix="[TEST]")

    return test_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _infer_last_conv_layer(model: keras.Model) -> str | None:
    """Return the name of the last Conv2D layer in the encoder (CNN) or None."""
    last_name = None
    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            last_name = layer.name
        elif hasattr(layer, 'layers'):
            for sub in layer.layers:
                if isinstance(sub, layers.Conv2D):
                    last_name = sub.name
    return last_name


def _print_metrics(m: dict[str, Any], prefix: str = "") -> None:
    keys = ['accuracy', 'roc_auc', 'sensitivity', 'specificity', 'ppv', 'npv']
    vals = '  '.join(f"{k}={m[k]:.4f}" for k in keys if k in m)
    print(f"{prefix}  {vals}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scratch supervised benchmark (no pretraining, no regularizer)"
    )
    p.add_argument('--data_root',    required=True, help='clean_ver_for_train path')
    p.add_argument('--encoder',      default='vit',
                   choices=['vit', 'swin', 'convnext', 'efficientnet'])
    p.add_argument('--img_size',     type=int, default=224)
    p.add_argument('--batch_size',   type=int, default=32)
    p.add_argument('--epochs',       type=int, default=100)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--output_dir',   default='outputs/benchmark')
    p.add_argument('--use_augmentation',  action='store_true')
    p.add_argument('--use_pretrained',    action='store_true',
                   help='CNN only — use ImageNet weights (ablation)')
    p.add_argument('--use_dropout',       action='store_true')
    p.add_argument('--use_weight_decay',  action='store_true')
    p.add_argument('--viz_every',    type=int, default=1,
                   help='Run viz callback every N epochs')
    p.add_argument('--seed',         type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    result = run_supervised_benchmark(
        data_root        = args.data_root,
        encoder_name     = args.encoder,
        input_shape      = (args.img_size, args.img_size, 3),
        batch_size       = args.batch_size,
        epochs           = args.epochs,
        lr               = args.lr,
        use_augmentation = args.use_augmentation,
        use_pretrained   = args.use_pretrained,
        use_dropout      = args.use_dropout,
        use_weight_decay = args.use_weight_decay,
        output_dir       = args.output_dir,
        viz_every_n_epochs = args.viz_every,
        seed             = args.seed,
    )
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('fpr', 'tpr')}, indent=2))
