"""conv_hybrid_vit.py

EfficientNetV2S → Conv1x1 projection → CLS token prepend → Shallow Transformer

Shape contract (512×512 input, stride-32 backbone)
---------------------------------------------------
  Input       : [B, 512, 512, 3]
  CNN fmap    : [B, 16, 16, 2048]   (ResNet50V2 output stride=32)
  Proj fmap   : [B, 16, 16, d_model]
  Tokens      : [B, 256, d_model]   flatten
  + CLS       : [B, 257, d_model]
  + Learnable PE : applied to patch tokens only (shape [1, n_patches, d_model])
  Transformer : n_layers × TransformerEncoderBlock
  Output dict:
    'embedding'          : [B, d_model]          <- CLS token (VICReg loss 용)
    'patch_tokens'       : [B, n_patches, d_model]
    'encoded_patches'    : [B, seq_len, d_model]  <- CLS + patches (legacy key)
    'attention_weights'  : list[ [B, n_heads, seq_len, seq_len] ] × n_layers
    'feature_map'        : [B, gh, gw, d_model]   <- projected fmap (vis 용)
    'patch_grid_size'    : [gh, gw]               <- e.g. [16, 16]
"""
from __future__ import annotations

import math
from typing import Any

import keras
import tensorflow as tf
from keras import layers


# ──────────────────────────────────────────────────────────────────────────────
# 1. Learnable Positional Embedding (bicubic, variable resolution)
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class OptionalAbsolutePositionalEmbedding(layers.Layer):
    """Learnable absolute positional embedding with bicubic interpolation.

    Stored as a [1, base_gh, base_gw, embed_dim] grid, bicubic-resized at
    runtime to actual (gh, gw). CLS token gets its own learnable embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        base_grid_size: tuple[int, int] = (12, 12),
        use_cls_token: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.base_grid_size = tuple(base_grid_size)
        self.use_cls_token = bool(use_cls_token)

    def build(self, input_shape):
        gh, gw = self.base_grid_size
        self.pos_grid = self.add_weight(
            shape=(1, gh, gw, self.embed_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="pos_grid",
        )
        if self.use_cls_token:
            self.cls_pos = self.add_weight(
                shape=(1, 1, self.embed_dim),
                initializer="zeros",
                trainable=True,
                name="cls_pos",
            )
        else:
            self.cls_pos = None
        super().build(input_shape)

    def call(
        self, x: tf.Tensor, grid_hw: tuple[tf.Tensor, tf.Tensor]
    ) -> tf.Tensor:
        gh, gw = grid_hw
        pos = tf.image.resize(self.pos_grid, size=(gh, gw), method="bicubic")
        pos = tf.reshape(pos, [1, gh * gw, self.embed_dim])
        if self.use_cls_token:
            cls_pos = tf.cast(self.cls_pos, x.dtype)
            pos = tf.concat([cls_pos, tf.cast(pos, x.dtype)], axis=1)  # [1, 1+N, D]
        else:
            pos = tf.cast(pos, x.dtype)
        return x + pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            embed_dim=self.embed_dim,
            base_grid_size=self.base_grid_size,
            use_cls_token=self.use_cls_token,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 2. TransformerEncoderBlock  (always returns attention scores)
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class TransformerEncoderBlock(layers.Layer):
    """Pre-LN Transformer block.  Always collects attention weights."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.mlp_dim   = int(mlp_dim)
        self.dropout   = float(dropout)

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn  = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
        )
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp   = keras.Sequential(
            [
                layers.Dense(mlp_dim, activation="gelu"),
                layers.Dropout(dropout),
                layers.Dense(embed_dim),
                layers.Dropout(dropout),
            ],
            name="ffn",
        )

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        return_attention: bool = True,
    ):
        y = self.norm1(x)
        attn_out, attn_scores = self.attn(
            y, y, return_attention_scores=True, training=training
        )  # attn_scores: [B, n_heads, seq, seq]
        x = x + self.drop1(attn_out, training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        if return_attention:
            return x, attn_scores
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 3. ConvPatchEmbedding  — CNN → Conv1x1 proj → flatten → CLS → learnable PE
# ────────────────────────────────────────��─────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class ConvPatchEmbedding(layers.Layer):
    """CNN-based patch tokenizer.

    CNN backbone (output stride 32)
      → Conv1x1 bottleneck → d_model
      → flatten patches [B, n_patches, d_model]
      → prepend CLS token  → [B, seq_len, d_model]  (seq_len = 1 + n_patches)
      → learnable PE [1, n_patches, d_model] added to patch tokens only
    """

    BACKBONE_MAP = {
        "efficientnetv2_s": (keras.applications.EfficientNetV2S, 1280),
        "efficientnetv2_m": (keras.applications.EfficientNetV2M, 1280),
        "resnet50v2":       (keras.applications.ResNet50V2,       2048),
        "densenet121":      (keras.applications.DenseNet121,      1024),
        "convnext_small":   (keras.applications.ConvNeXtSmall,    768),
        "convnext_tiny":    (keras.applications.ConvNeXtTiny,     768),
    }

    def __init__(
        self,
        backbone_name: str = "efficientnetv2_s",
        d_model: int = 256,
        n_patches: int = 256,           # cfg.n_patches  (e.g. 16*16=256 for res=512)
        imagenet_pretrained: bool = True,
        backbone_trainable: bool = False,
        # kept for backward-compat; ignored (learnable PE is always used)
        use_sinusoidal_pe: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name       = backbone_name
        self.d_model             = int(d_model)
        self.n_patches           = int(n_patches)
        self.imagenet_pretrained = bool(imagenet_pretrained)
        self.backbone_trainable  = bool(backbone_trainable)
        self.use_sinusoidal_pe   = use_sinusoidal_pe  # stored only for get_config compat

        if backbone_name not in self.BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        BackboneCls, _ = self.BACKBONE_MAP[backbone_name]

        self._cnn = BackboneCls(
            include_top=False,
            weights="imagenet" if imagenet_pretrained else None,
        )
        self._cnn.trainable = backbone_trainable

        self._proj      = layers.Conv2D(d_model, kernel_size=1, use_bias=False, name="conv_proj")
        self._proj_norm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        # CLS token
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
        # Learnable PE for patch tokens only: [1, n_patches, d_model]
        self.pos_embed = self.add_weight(
            shape=(1, self.n_patches, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="pos_embed",
        )
        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Returns
        -------
        tokens  : [B, seq_len, d_model]   (seq_len = 1 + n_patches)
        fmap    : [B, gh, gw, d_model]
        gh, gw  : scalar tensors
        """
        fmap_raw = self._cnn(x, training=(training and self.backbone_trainable))
        fmap     = self._proj(fmap_raw, training=training)
        fmap     = self._proj_norm(fmap, training=training)

        B  = tf.shape(x)[0]
        gh = tf.shape(fmap)[1]
        gw = tf.shape(fmap)[2]
        N  = gh * gw  # == self.n_patches at runtime

        # Flatten: [B, N, d_model]
        patches = tf.reshape(fmap, [B, N, self.d_model])

        # Learnable PE — pos_embed is [1, n_patches, d_model], cast for mixed-precision
        patches = patches + tf.cast(self.pos_embed, patches.dtype)

        # Prepend CLS: [B, seq_len, d_model]
        cls    = tf.cast(tf.tile(self.cls_token, [B, 1, 1]), patches.dtype)
        tokens = tf.concat([cls, patches], axis=1)

        return tokens, fmap, gh, gw

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            backbone_name=self.backbone_name,
            d_model=self.d_model,
            n_patches=self.n_patches,
            imagenet_pretrained=self.imagenet_pretrained,
            backbone_trainable=self.backbone_trainable,
            use_sinusoidal_pe=self.use_sinusoidal_pe,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 4. ConvHybridViTBackbone  — full encoder
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class ConvHybridViTBackbone(keras.Model):
    """CNN backbone + Shallow Transformer with CLS token.

    Output dict
    -----------
    'embedding'          : [B, d_model]                  <- CLS (VICReg용)
    'patch_tokens'       : [B, n_patches, d_model]
    'encoded_patches'    : [B, seq_len, d_model]         <- CLS + patches
    'attention_weights'  : list of [B, n_heads, seq_len, seq_len]
    'feature_map'        : [B, gh, gw, d_model]
    'patch_grid_size'    : [gh, gw]
    'gap_vector'         : alias for 'embedding'
    """

    def __init__(
        self,
        input_shape: tuple = (512, 512, 3),
        backbone_name: str = "resnet50v2",
        d_model: int = 256,
        n_patches: int = 256,           # cfg.n_patches
        depth: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        imagenet_pretrained: bool = True,
        backbone_trainable: bool = False,
        use_sinusoidal_pe: bool = True,  # kept for compat; ignored
        # Legacy compat params
        patch_size: int = 24,
        embed_dim: int = 256,
        conv_stem_depth: int = 1,
        stem_kernel_size: int = 3,
        use_positional_encoding: bool = True,
        base_grid_size: tuple = (12, 12),
        pool_mode: str = "cls",
        name: str = "conv_hybrid_vit_backbone",
    ):
        super().__init__(name=name)
        self.input_spec = keras.layers.InputSpec(ndim=4, axes={-1: input_shape[-1]})
        self.d_model             = int(d_model)
        self.n_patches           = int(n_patches)
        self.depth               = int(depth)
        self.num_heads           = int(num_heads)
        self.mlp_dim             = int(mlp_dim)
        self.dropout_rate        = float(dropout)
        self.backbone_name       = backbone_name
        self.imagenet_pretrained = bool(imagenet_pretrained)
        self.backbone_trainable  = bool(backbone_trainable)
        self.use_sinusoidal_pe   = use_sinusoidal_pe
        self.pool_mode           = pool_mode

        self.patch_embed = ConvPatchEmbedding(
            backbone_name=backbone_name,
            d_model=d_model,
            n_patches=n_patches,           # ← cfg.n_patches passed here
            imagenet_pretrained=imagenet_pretrained,
            backbone_trainable=backbone_trainable,
            use_sinusoidal_pe=use_sinusoidal_pe,
            name="patch_embed",
        )
        self.blocks = [
            TransformerEncoderBlock(
                embed_dim=d_model,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
                name=f"block_{i}",
            )
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6, name="norm")

    def call(
        self,
        x: tf.Tensor,
        training: bool = False,
        return_attention: bool = True,
    ) -> dict[str, Any]:
        tokens, fmap, gh, gw = self.patch_embed(x, training=training)

        y = tokens
        attn_all = []
        for block in self.blocks:
            y, attn = block(y, training=training, return_attention=True)
            attn_all.append(attn)

        y = self.norm(y)

        cls_token    = y[:, 0, :]
        patch_tokens = y[:, 1:, :]

        embedding = cls_token if self.pool_mode == "cls" else tf.reduce_mean(patch_tokens, axis=1)

        return {
            "embedding"         : embedding,
            "gap_vector"        : embedding,
            "patch_tokens"      : patch_tokens,
            "encoded_patches"   : y,
            "attention_weights" : attn_all,
            "last_encoder_layer_attentional_weights": attn_all[-1] if attn_all else None,
            "feature_map"       : fmap,
            "patch_grid_size"   : tf.stack([gh, gw]),
        }

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            backbone_name=self.backbone_name,
            d_model=self.d_model,
            n_patches=self.n_patches,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate,
            imagenet_pretrained=self.imagenet_pretrained,
            backbone_trainable=self.backbone_trainable,
            use_sinusoidal_pe=self.use_sinusoidal_pe,
            pool_mode=self.pool_mode,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 5. Factory
# ──────────────────────────────────────────────────────────────────────────────
def build_conv_hybrid_vit(
    input_shape: tuple = (512, 512, 3),
    backbone_name: str = "resnet50v2",
    d_model: int = 256,
    n_patches: int = 256,           # cfg.n_patches
    depth: int = 2,
    num_heads: int = 8,
    mlp_dim: int = 512,
    dropout: float = 0.1,
    imagenet_pretrained: bool = True,
    backbone_trainable: bool = False,
    use_sinusoidal_pe: bool = True,
    pool_mode: str = "cls",
) -> ConvHybridViTBackbone:
    """Build and warm-start ConvHybridViTBackbone."""
    model = ConvHybridViTBackbone(
        input_shape=input_shape,
        backbone_name=backbone_name,
        d_model=d_model,
        n_patches=n_patches,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        imagenet_pretrained=imagenet_pretrained,
        backbone_trainable=backbone_trainable,
        use_sinusoidal_pe=use_sinusoidal_pe,
        pool_mode=pool_mode,
    )
    dummy = tf.zeros((2, *input_shape))
    _ = model(dummy, training=False)
    h, w, _ = input_shape
    gh, gw  = h // 32, w // 32
    n_patch = gh * gw
    print(
        f"[ConvHybridViT] backbone={backbone_name}  "
        f"input={input_shape}  fmap={gh}×{gw}  n_patch={n_patch}  "
        f"seq_len=1+{n_patch}={1+n_patch}  d_model={d_model}  "
        f"depth={depth}  n_heads={num_heads}  PE=learnable"
    )
    print(
        f"[ConvHybridViT] total_params={model.count_params():,}  "
        f"trainable={sum(tf.size(v).numpy() for v in model.trainable_variables):,}"
    )
    return model
