from __future__ import annotations

from typing import Any

import keras
import tensorflow as tf
from keras import layers


class OptionalAbsolutePositionalEmbedding(layers.Layer):
    """Learnable absolute positional embedding for variable token lengths.

    Stores a base grid and interpolates it at runtime to the current token grid.
    If use_positional_encoding=False, this layer is skipped by the backbone.
    """

    def __init__(
        self,
        embed_dim: int,
        base_grid_size: tuple[int, int] = (24, 24),
        use_cls_token: bool = False,
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
            initializer=keras.initializers.TruncatedNormal(mean=0.0, stddev=0.02),
            trainable=True,
            name='pos_grid',
        )
        if self.use_cls_token:
            self.cls_pos = self.add_weight(
                shape=(1, 1, self.embed_dim),
                initializer='zeros',
                trainable=True,
                name='cls_pos',
            )
        else:
            self.cls_pos = None
        super().build(input_shape)

    def call(self, x: tf.Tensor, grid_hw: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        gh, gw = grid_hw
        pos = tf.image.resize(self.pos_grid, size=(gh, gw), method='bicubic')
        pos = tf.reshape(pos, [1, gh * gw, self.embed_dim])
        if self.use_cls_token:
            cls_pos = tf.cast(self.cls_pos, x.dtype)
            pos = tf.concat([cls_pos, tf.cast(pos, x.dtype)], axis=1)
        else:
            pos = tf.cast(pos, x.dtype)
        return x + pos


class LightweightInvertedBottleneck(layers.Layer):
    """Lightweight inverted bottleneck stem before patch projection.

    Applies a cheap local feature extractor at full resolution without OOM:
        PW expand (1x1) -> DW conv (3x3) -> PW shrink (1x1)

    in_channels=3, expand_ratio=8 -> hidden=24ch.
    Memory at (B=16, 384x384): 16 * 384 * 384 * 24 * 4 ~ 0.23 GB.
    No residual (in_channels=3 is RGB; residual would add no benefit here).

    Shape:
        input : (B, H, W, in_channels)
        output: (B, H, W, in_channels)   # shape preserved
    """

    def __init__(
        self,
        in_channels: int = 3,
        expand_ratio: int = 8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        hidden_dim = in_channels * expand_ratio  # 3 * 8 = 24

        # Step 1: pointwise expand — channel mixing, no spatial op
        self.pw_expand = keras.Sequential([
            layers.Conv2D(hidden_dim, kernel_size=1, use_bias=False, padding='same'),
            layers.BatchNormalization(),
            layers.Activation('gelu'),
        ], name='pw_expand')

        # Step 2: depthwise 3x3 — spatial feature extraction, cheap
        self.dw_conv = keras.Sequential([
            layers.DepthwiseConv2D(kernel_size=3, padding='same', use_bias=False),
            layers.BatchNormalization(),
            layers.Activation('gelu'),
        ], name='dw_conv')

        # Step 3: pointwise shrink — project back, NO activation (information preservation)
        self.pw_shrink = keras.Sequential([
            layers.Conv2D(in_channels, kernel_size=1, use_bias=False, padding='same'),
            layers.BatchNormalization(),
        ], name='pw_shrink')

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # (B,H,W,3) -> expand(24) -> dw(24) -> shrink(3) -> (B,H,W,3)
        out = self.pw_expand(x,   training=training)
        out = self.dw_conv(out,   training=training)
        out = self.pw_shrink(out, training=training)
        return out


class ConvTokenEmbedding(layers.Layer):
    """Convolutional patch embedding with optional inverted bottleneck stem.

    Pipeline:
        LightweightInvertedBottleneck (in_ch=3, expand=8)  [shape preserved]
        -> Conv2D patch_proj (kernel=patch_size, stride=patch_size)
        -> reshape to tokens

    Shape contract:
        input : (B, H, W, 3)
        output: tokens (B, N, embed_dim), fmap (B, H/p, W/p, embed_dim), gh, gw
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        conv_stem_depth: int = 1,
        stem_kernel_size: int = 3,
        stem_activation: str = 'gelu',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.conv_stem_depth = int(conv_stem_depth)
        self.stem_kernel_size = int(stem_kernel_size)
        self.stem_activation = stem_activation

        # Inverted bottleneck stem: in_channels=3, expand_ratio=8 -> hidden=24ch
        self.inv_bottleneck = LightweightInvertedBottleneck(
            in_channels=3,
            expand_ratio=8,
            name='inv_bottleneck_stem',
        )

        self.patch_proj = layers.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding='valid',
            use_bias=True,
            name='patch_proj',
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # (B, H, W, 3) -> inv_bottleneck -> (B, H, W, 3) -> patch_proj -> (B, H/p, W/p, D)
        x = tf.cast(x, tf.float32)
        x = self.inv_bottleneck(x, training=training)   # local feature extraction
        fmap = self.patch_proj(x)                        # patch tokenisation
        gh = tf.shape(fmap)[1]
        gw = tf.shape(fmap)[2]
        c  = tf.shape(fmap)[3]
        tokens = tf.reshape(fmap, [tf.shape(fmap)[0], gh * gw, c])
        return tokens, fmap, gh, gw


class TransformerEncoderBlock(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
        )
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = keras.Sequential([
            layers.Dense(mlp_dim, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(embed_dim),
            layers.Dropout(dropout),
        ])

    def call(self, x: tf.Tensor, training: bool = False, return_attention: bool = False):
        y = self.norm1(x)
        if return_attention:
            attn_out, attn_scores = self.attn(y, y, return_attention_scores=True, training=training)
        else:
            attn_out = self.attn(y, y, training=training)
            attn_scores = None
        x = x + self.drop1(attn_out, training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        if return_attention:
            return x, attn_scores
        return x


class ConvHybridViTBackbone(keras.Model):
    """Variable-resolution Conv-Hybrid ViT backbone.

    Shape contract
    --------------
    Input : [B, H, W, 3], with H and W multiples of patch_size.
    Stem  : LightweightInvertedBottleneck(in_ch=3, expand=8) -> [B, H, W, 3]
    Patch : Conv2D(stride=patch_size) -> [B, H/p, W/p, embed_dim]
    Token : reshape -> [B, N, embed_dim], N = (H/p) * (W/p)

    Positional encoding
    -------------------
    - use_positional_encoding=False: no positional encoding
    - use_positional_encoding=True : interpolated learnable absolute 2D embedding
                                     initialised with TruncatedNormal(0, 0.02)

    This backbone is designed to accept multiple resolutions with a single set of weights.
    """

    def __init__(
        self,
        input_shape: tuple[int | None, int | None, int] = (None, None, 3),
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        conv_stem_depth: int = 1,
        stem_kernel_size: int = 3,
        use_positional_encoding: bool = False,
        base_grid_size: tuple[int, int] = (24, 24),
        pool_mode: str = 'gap',
        name: str = 'conv_hybrid_vit_backbone',
    ):
        super().__init__(name=name)
        self.input_spec = keras.layers.InputSpec(ndim=4, axes={-1: input_shape[-1]})
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.pool_mode = pool_mode

        self.token_embed = ConvTokenEmbedding(
            patch_size=patch_size,
            embed_dim=embed_dim,
            conv_stem_depth=conv_stem_depth,
            stem_kernel_size=stem_kernel_size,
            name='token_embed',
        )
        self.pos_embed = OptionalAbsolutePositionalEmbedding(
            embed_dim=embed_dim,
            base_grid_size=base_grid_size,
            use_cls_token=False,
            name='pos_embed',
        )
        self.blocks = [
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout, name=f'block_{i}')
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False, return_attention: bool = True) -> dict[str, Any]:
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        tf.debugging.assert_equal(
            tf.math.floormod(h, self.patch_size),
            0,
            message='Input height must be divisible by patch_size.',
        )
        tf.debugging.assert_equal(
            tf.math.floormod(w, self.patch_size),
            0,
            message='Input width must be divisible by patch_size.',
        )

        tokens, fmap, gh, gw = self.token_embed(x, training=training)
        if self.use_positional_encoding:
            tokens = self.pos_embed(tokens, grid_hw=(gh, gw))

        attn_all = []
        y = tokens
        for i, block in enumerate(self.blocks):
            ret_attn = return_attention and (i == len(self.blocks) - 1)
            if ret_attn:
                y, attn = block(y, training=training, return_attention=True)
                attn_all.append(attn)
            else:
                y = block(y, training=training, return_attention=False)
        y = self.norm(y)

        if self.pool_mode == 'gap':
            embedding = tf.reduce_mean(y, axis=1)
        elif self.pool_mode == 'max':
            embedding = tf.reduce_max(y, axis=1)
        else:
            raise ValueError(f'Unsupported pool_mode: {self.pool_mode}')

        return {
            'encoded_patches': y,
            'feature_map': fmap,
            'patch_grid_size': tf.stack([gh, gw]),
            'last_encoder_layer_attentional_weights': attn_all[-1] if attn_all else None,
            'attention_weights': attn_all,
            'embedding': embedding,
            'gap_vector': embedding,
        }
