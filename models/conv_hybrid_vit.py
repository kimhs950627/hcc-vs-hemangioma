from __future__ import annotations

from typing import Any

import keras
import tensorflow as tf
from keras import layers


class OptionalAbsolutePositionalEmbedding(layers.Layer):
    """Learnable absolute positional embedding with bicubic interpolation.

    Stores a base grid (1, base_gh, base_gw, embed_dim) and bicubic-resizes
    it at runtime to the actual patch grid (gh, gw).
    Supports arbitrary input resolutions without shape errors.
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


class ConvTokenEmbedding(layers.Layer):
    """Patch embedding via extract_patches + Dense (no Conv2D).

    Using tf.image.extract_patches instead of Conv2D eliminates the
    spatial inductive bias that causes geographic activation patterns.
    Mathematically equivalent to stride-p Conv2D but with uniform
    treatment of all patch positions.

    Shape contract:
        input : (B, H, W, 3)
        output: tokens (B, N, embed_dim), fmap_placeholder None, gh, gw
                N = (H // patch_size) * (W // patch_size)
    """

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 384,
        conv_stem_depth: int = 1,       # kept for API compat, unused
        stem_kernel_size: int = 3,      # kept for API compat, unused
        stem_activation: str = 'gelu',  # kept for API compat, unused
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.embed_dim  = int(embed_dim)
        # Dense projection: weight (p*p*3, embed_dim) = (768, 384)
        # Param count identical to Conv2D(384, 16, stride=16): 16*16*3*384 = 294,912
        self.patch_proj = layers.Dense(embed_dim, use_bias=True, name='patch_proj')

    def call(self, x: tf.Tensor, training: bool = False) -> tuple[tf.Tensor, None, tf.Tensor, tf.Tensor]:
        x = tf.cast(x, tf.float32)
        p = self.patch_size

        # (B, H, W, 3) → (B, gh, gw, p*p*3)
        patches = tf.image.extract_patches(
            images=x,
            sizes=[1, p, p, 1],
            strides=[1, p, p, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        gh = tf.shape(patches)[1]
        gw = tf.shape(patches)[2]

        # (B, gh, gw, p*p*3) → (B, gh*gw, p*p*3) → Dense → (B, N, embed_dim)
        B       = tf.shape(x)[0]
        flat    = tf.reshape(patches, [B, gh * gw, p * p * 3])
        tokens  = self.patch_proj(flat)   # (B, N, embed_dim)

        # fmap=None: ConvHybridViTBackbone uses it only for visualization;
        # callers must guard with `if fmap is not None`
        return tokens, None, gh, gw


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
    """Variable-resolution ViT backbone (no Conv2D in patch embedding).

    Shape contract
    --------------
    Input : [B, H, W, 3], H and W multiples of patch_size.
    Patch : extract_patches -> Dense -> [B, N, embed_dim]
            N = (H/p) * (W/p)
    PE    : OptionalAbsolutePositionalEmbedding (bicubic, variable resolution)

    No spatial inductive bias from Conv2D.
    Param count of patch_proj identical to the former Conv2D (294,912).
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
        self.embed_dim  = int(embed_dim)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.pool_mode  = pool_mode

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
            tf.math.floormod(h, self.patch_size), 0,
            message='Input height must be divisible by patch_size.',
        )
        tf.debugging.assert_equal(
            tf.math.floormod(w, self.patch_size), 0,
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
            'feature_map': fmap,   # None (kept for API compat)
            'patch_grid_size': tf.stack([gh, gw]),
            'last_encoder_layer_attentional_weights': attn_all[-1] if attn_all else None,
            'attention_weights': attn_all,
            'embedding': embedding,
            'gap_vector': embedding,
        }
