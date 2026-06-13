"""conv_hybrid_vit.py

EfficientNetV2S → Conv1x1 projection → CLS token prepend → Shallow Transformer

Shape contract (384×384 input, stride-32 backbone)
---------------------------------------------------
  Input       : [B, 384, 384, 3]
  CNN fmap    : [B, 12, 12, 1280]   (EfficientNetV2S output stride=32)
  Proj fmap   : [B, 12, 12, d_model]
  Tokens      : [B, 144, d_model]   flatten
  + CLS       : [B, 145, d_model]
  + 2D PE     : same shape
  Transformer : n_layers × TransformerEncoderBlock
  Output dict:
    'embedding'          : [B, d_model]          <- CLS token (VICReg loss 용)
    'patch_tokens'       : [B, 144, d_model]
    'encoded_patches'    : [B, 145, d_model]      <- CLS + patches (legacy key)
    'attention_weights'  : list[ [B, n_heads, 145, 145] ] × n_layers
    'feature_map'        : [B, 12, 12, d_model]   <- projected fmap (vis 용)
    'patch_grid_size'    : [gh, gw]               <- e.g. [12, 12]
"""
from __future__ import annotations

import math
from typing import Any

import keras
import tensorflow as tf
from keras import layers


# ──────────────────────────────────────────────────────────────────────────────
# 1. 2D Sinusoidal Positional Encoding
# ──────────────────────────────────────────────────────────────────────────────
def _build_2d_sinusoidal_pe(
    gh: int, gw: int, embed_dim: int, dtype=tf.float32
) -> tf.Tensor:
    """Return 2-D sinusoidal PE of shape [1, gh*gw, embed_dim].

    Half dims for row, half for col — interleaved sin/cos pairs.
    CLS token PE is handled separately (zeros).
    """
    assert embed_dim % 4 == 0, "embed_dim must be divisible by 4 for 2D PE"
    half = embed_dim // 2  # used for rows & cols separately

    def _1d_sincos(length: int, dim: int) -> tf.Tensor:
        """[length, dim] sinusoidal encoding."""
        positions = tf.cast(tf.range(length), dtype)[:, None]  # [L, 1]
        dims = tf.cast(tf.range(0, dim, 2), dtype)[None, :]    # [1, dim/2]
        theta = positions / tf.pow(10000.0, dims / tf.cast(dim, dtype))
        sin = tf.math.sin(theta)  # [L, dim/2]
        cos = tf.math.cos(theta)  # [L, dim/2]
        # interleave: [L, dim]
        return tf.reshape(tf.stack([sin, cos], axis=-1), [length, dim])

    row_pe = _1d_sincos(gh, half)  # [gh, half]
    col_pe = _1d_sincos(gw, half)  # [gw, half]

    # Broadcast to [gh, gw, embed_dim]
    row_pe = tf.tile(row_pe[:, None, :], [1, gw, 1])  # [gh, gw, half]
    col_pe = tf.tile(col_pe[None, :, :], [gh, 1, 1])  # [gh, gw, half]
    pe_2d = tf.concat([row_pe, col_pe], axis=-1)       # [gh, gw, embed_dim]

    pe_flat = tf.reshape(pe_2d, [1, gh * gw, embed_dim])  # [1, N, D]
    return tf.cast(pe_flat, dtype)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Learnable Positional Embedding (bicubic, variable resolution)
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
# 3. TransformerEncoderBlock  (always returns attention scores)
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
# 4. ConvPatchEmbedding  — EfficientNetV2S → Conv1x1 proj → flatten → CLS
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class ConvPatchEmbedding(layers.Layer):
    """CNN-based patch tokenizer.

    EfficientNetV2S (output stride 32, ImageNet-pretrained)
      → Conv1x1 bottleneck (1280 → d_model)
      → flatten patches [B, N, d_model]
      → prepend CLS token  → [B, 1+N, d_model]
      → 2D sinusoidal PE on patches (CLS gets zeros)
    """

    BACKBONE_MAP = {
        "efficientnetv2_s": (keras.applications.EfficientNetV2S, 1280),
        "efficientnetv2_m": (keras.applications.EfficientNetV2M, 1280),
        "resnet50v2":       (keras.applications.ResNet50V2,       2048),
        "densenet121":      (keras.applications.DenseNet121,      1024),
        "convnext_small":   (keras.applications.ConvNeXtSmall,    768),
    }

    def __init__(
        self,
        backbone_name: str = "efficientnetv2_s",
        d_model: int = 256,
        imagenet_pretrained: bool = True,
        backbone_trainable: bool = False,
        use_sinusoidal_pe: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name       = backbone_name
        self.d_model             = int(d_model)
        self.imagenet_pretrained = bool(imagenet_pretrained)
        self.backbone_trainable  = bool(backbone_trainable)
        self.use_sinusoidal_pe   = bool(use_sinusoidal_pe)

        if backbone_name not in self.BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        BackboneCls, _ = self.BACKBONE_MAP[backbone_name]

        # Build CNN backbone
        self._cnn = BackboneCls(
            include_top=False,
            weights="imagenet" if imagenet_pretrained else None,
        )
        self._cnn.trainable = backbone_trainable

        # 1×1 projection
        self._proj = layers.Conv2D(
            d_model, kernel_size=1, use_bias=False, name="conv_proj"
        )
        self._proj_norm = layers.LayerNormalization(epsilon=1e-6)

        # Learnable CLS token  [1, 1, d_model]
        # Created in build() because we need to register as weight

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="cls_token",
        )
        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Returns
        -------
        tokens  : [B, 1+N, d_model]   CLS + patches
        fmap    : [B, gh, gw, d_model] projected feature map (for vis)
        gh, gw  : scalar tensors
        """
        # CNN
        fmap_raw = self._cnn(x, training=(training and self.backbone_trainable))
        # [B, gh, gw, C_backbone]

        # Project
        fmap = self._proj(fmap_raw, training=training)   # [B, gh, gw, d_model]
        fmap = self._proj_norm(fmap, training=training)

        B  = tf.shape(x)[0]
        gh = tf.shape(fmap)[1]
        gw = tf.shape(fmap)[2]
        N  = gh * gw

        # Flatten patches: [B, N, d_model]
        patches = tf.reshape(fmap, [B, N, self.d_model])

        # 2D sinusoidal PE on patches
        if self.use_sinusoidal_pe:
            # Static PE at fixed grid size for speed (built dynamically if needed)
            pe = _build_2d_sinusoidal_pe(
                tf.get_static_value(gh) or 12,
                tf.get_static_value(gw) or 12,
                self.d_model,
                dtype=patches.dtype,
            )  # [1, N, d_model]
            patches = patches + pe

        # Prepend CLS
        cls = tf.cast(
            tf.tile(self.cls_token, [B, 1, 1]), patches.dtype
        )  # [B, 1, d_model]
        tokens = tf.concat([cls, patches], axis=1)  # [B, 1+N, d_model]

        return tokens, fmap, gh, gw

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            backbone_name=self.backbone_name,
            d_model=self.d_model,
            imagenet_pretrained=self.imagenet_pretrained,
            backbone_trainable=self.backbone_trainable,
            use_sinusoidal_pe=self.use_sinusoidal_pe,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 5. ConvHybridViTBackbone  — full encoder
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class ConvHybridViTBackbone(keras.Model):
    """EfficientNetV2S + Shallow Transformer with CLS token.

    Output dict
    -----------
    'embedding'          : [B, d_model]                  <- CLS (VICReg용)
    'patch_tokens'       : [B, N, d_model]               <- patch repr
    'encoded_patches'    : [B, 1+N, d_model]             <- CLS + patches
    'attention_weights'  : list of [B, n_heads, 1+N, 1+N]  (all layers)
    'feature_map'        : [B, gh, gw, d_model]          <- projected fmap
    'patch_grid_size'    : [gh, gw]
    'gap_vector'         : alias for 'embedding'
    """

    def __init__(
        self,
        input_shape: tuple = (384, 384, 3),
        backbone_name: str = "efficientnetv2_s",
        d_model: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        imagenet_pretrained: bool = True,
        backbone_trainable: bool = False,
        use_sinusoidal_pe: bool = True,
        # Legacy compat params (ignored but accepted)
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
        self.depth               = int(depth)
        self.num_heads           = int(num_heads)
        self.mlp_dim             = int(mlp_dim)
        self.dropout_rate        = float(dropout)
        self.backbone_name       = backbone_name
        self.imagenet_pretrained = bool(imagenet_pretrained)
        self.backbone_trainable  = bool(backbone_trainable)
        self.use_sinusoidal_pe   = bool(use_sinusoidal_pe)
        self.pool_mode           = pool_mode  # 'cls' | 'gap'

        self.patch_embed = ConvPatchEmbedding(
            backbone_name=backbone_name,
            d_model=d_model,
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
        # 1. Patch embedding + CLS prepend
        tokens, fmap, gh, gw = self.patch_embed(x, training=training)
        # tokens: [B, 1+N, d_model]

        # 2. Transformer blocks — collect all attention weights
        y = tokens
        attn_all = []
        for block in self.blocks:
            y, attn = block(y, training=training, return_attention=True)
            attn_all.append(attn)  # [B, n_heads, 1+N, 1+N]

        y = self.norm(y)  # [B, 1+N, d_model]

        # 3. Extract CLS and patch tokens
        cls_token   = y[:, 0, :]    # [B, d_model]
        patch_tokens = y[:, 1:, :]  # [B, N, d_model]

        # 4. Pooling mode
        if self.pool_mode == "cls":
            embedding = cls_token
        else:  # 'gap'
            embedding = tf.reduce_mean(patch_tokens, axis=1)

        return {
            "embedding"          : embedding,          # [B, d_model]   <- VICReg 용
            "gap_vector"         : embedding,          # alias
            "patch_tokens"       : patch_tokens,       # [B, N, d_model]
            "encoded_patches"    : y,                  # [B, 1+N, d_model]
            "attention_weights"  : attn_all,           # list of [B, H, 1+N, 1+N]
            "last_encoder_layer_attentional_weights": attn_all[-1] if attn_all else None,
            "feature_map"        : fmap,               # [B, gh, gw, d_model]
            "patch_grid_size"    : tf.stack([gh, gw]),
        }

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            backbone_name=self.backbone_name,
            d_model=self.d_model,
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
# 6. Factory
# ──────────────────────────────────────────────────────────────────────────────
def build_conv_hybrid_vit(
    input_shape: tuple = (384, 384, 3),
    backbone_name: str = "efficientnetv2_s",
    d_model: int = 256,
    depth: int = 2,
    num_heads: int = 8,
    mlp_dim: int = 512,
    dropout: float = 0.1,
    imagenet_pretrained: bool = True,
    backbone_trainable: bool = False,
    use_sinusoidal_pe: bool = True,
    pool_mode: str = "cls",
) -> ConvHybridViTBackbone:
    """Build and warm-start ConvHybridViTBackbone.

    Runs a dummy forward pass to initialize all weights.
    """
    model = ConvHybridViTBackbone(
        input_shape=input_shape,
        backbone_name=backbone_name,
        d_model=d_model,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        dropout=dropout,
        imagenet_pretrained=imagenet_pretrained,
        backbone_trainable=backbone_trainable,
        use_sinusoidal_pe=use_sinusoidal_pe,
        pool_mode=pool_mode,
    )
    # Warm-start
    dummy = tf.zeros((2, *input_shape))
    _ = model(dummy, training=False)
    h, w, _ = input_shape
    gh, gw  = h // 32, w // 32
    n_patch = gh * gw
    cnn_cls, cnn_ch = ConvPatchEmbedding.BACKBONE_MAP[backbone_name]
    print(
        f"[ConvHybridViT] backbone={backbone_name}  "
        f"input={input_shape}  fmap={gh}×{gw}  n_patch={n_patch}  "
        f"seq_len=1+{n_patch}={1+n_patch}  d_model={d_model}  "
        f"depth={depth}  n_heads={num_heads}"
    )
    print(
        f"[ConvHybridViT] total_params={model.count_params():,}  "
        f"trainable={sum(tf.size(v).numpy() for v in model.trainable_variables):,}"
    )
    return model
