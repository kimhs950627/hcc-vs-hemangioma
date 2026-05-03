from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras
from keras import layers


# ---------------------------------------------------------------------------
# Patch Utilities
# ---------------------------------------------------------------------------

def patchify(images: tf.Tensor, patch_size: int = 16) -> tf.Tensor:
    """Extract non-overlapping patches from images for SimMIM target.

    Args:
        images     : (B, H, W, C) float32  in [0, 1]
        patch_size : ViT patch size (must divide H and W evenly)

    Returns:
        patches : (B, N, patch_size^2 * C)  where N = (H//P) * (W//P)
    """
    patches = tf.image.extract_patches(
        images=images,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID',
    )  # (B, n_h, n_w, patch_size^2 * C)
    B = tf.shape(images)[0]
    patch_dim = patch_size * patch_size * int(images.shape[-1])
    return tf.reshape(patches, [B, -1, patch_dim])  # (B, N, patch_dim)


# ---------------------------------------------------------------------------
# Core ViT Layers
# ---------------------------------------------------------------------------

class PatchExtract(layers.Layer):
    def __init__(self, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images: tf.Tensor) -> tf.Tensor:
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dim = tf.shape(patches)[-1]
        return tf.reshape(patches, [tf.shape(images)[0], -1, patch_dim])


class LearnableCLSToken(layers.Layer):
    def build(self, input_shape):
        d = int(input_shape[-1])
        self.cls = self.add_weight(
            shape=(1, 1, d), initializer='zeros', trainable=True, name='cls_token'
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        b = tf.shape(x)[0]
        cls = tf.repeat(self.cls, repeats=b, axis=0)
        return tf.concat([cls, x], axis=1)


class TrainablePositionalEmbedding(layers.Layer):
    """Learnable positional embedding with bilinear interpolation.

    At build time the embedding is created for the input length N_train.
    At runtime, if the sequence length N_run differs (e.g. local crops
    have fewer patches than the global view), the patch tokens are
    bilinearly interpolated assuming a square grid, so:

        N_train = h_t * w_t + 1   (1 CLS token)
        N_run   = h_r * w_r + 1

    The CLS token embedding is never interpolated — it is reused as-is.

    IMPORTANT: always warm-up the encoder with a canonical-resolution
    dummy forward pass *before* the first training step so that
    _h_built / _w_built reflect the intended resolution:

        dummy = tf.zeros([1, 224, 224, 3])
        _ = encoder(dummy, training=False)
    """

    def build(self, input_shape):
        n = int(input_shape[1])   # N_patches + 1 CLS
        d = int(input_shape[2])
        self.pos = self.add_weight(
            shape=(1, n, d), initializer='random_normal',
            trainable=True, name='pos_embed',
        )
        # Store the *patch* grid dimensions (excluding CLS token).
        # We derive h_t / w_t from the *static* stored weight shape so
        # that tf.reshape inside _interpolate_pos always gets constant
        # integers — this avoids the shape-mismatch crash when
        # the runtime sequence length differs from the build-time length.
        n_patches = n - 1  # exclude CLS
        h_t = int(round(n_patches ** 0.5))
        assert h_t * h_t == n_patches, (
            f"TrainablePositionalEmbedding: n_patches={n_patches} is not a "
            f"perfect square — cannot infer 2-D grid. "
            f"Make sure the encoder is first called with the canonical "
            f"square-resolution input (e.g. 224x224)."
        )
        self._h_built = h_t
        self._w_built = h_t  # square grid assumed
        self._n_train = n    # full sequence length incl. CLS

    def _interpolate_pos(self, n_run: int, d: int) -> tf.Tensor:
        """Bilinear-interpolate patch positional embeddings to n_run tokens.

        Args:
            n_run : target total sequence length (incl. CLS token)
            d     : embedding dimension

        Returns:
            pos : (1, n_run, d)
        """
        cls_pos   = self.pos[:, :1, :]   # (1, 1, D) — not interpolated
        patch_pos = self.pos[:, 1:, :]   # (1, h_t*w_t, D)

        # Use *static* h_t / w_t recorded at build time — never dynamic.
        h_t = self._h_built
        w_t = self._w_built

        # Target grid size from runtime sequence length
        n_patches_run = n_run - 1
        h_r = tf.cast(
            tf.math.round(tf.sqrt(tf.cast(n_patches_run, tf.float32))),
            tf.int32,
        )
        w_r = tf.cast(
            tf.math.ceil(
                tf.cast(n_patches_run, tf.float32) / tf.cast(h_r, tf.float32)
            ),
            tf.int32,
        )

        # (1, h_t*w_t, D) -> (1, h_t, w_t, D) -> bilinear -> (1, h_r*w_r, D)
        # h_t and w_t are Python ints so tf.reshape gets a constant shape.
        patch_pos_2d = tf.reshape(patch_pos, [1, h_t, w_t, d])
        patch_pos_2d = tf.image.resize(patch_pos_2d, [h_r, w_r], method='bilinear')
        patch_pos_1d = tf.reshape(patch_pos_2d, [1, h_r * w_r, d])

        return tf.concat([cls_pos, patch_pos_1d], axis=1)  # (1, n_run, D)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """Add positional embedding; interpolate if runtime length != build length."""
        n_run   = tf.shape(x)[1]         # runtime sequence length (dynamic)
        d       = tf.shape(x)[2]         # embedding dim (dynamic, used for resize)
        n_train = self._n_train          # build-time length (Python int)

        pos = tf.cond(
            tf.equal(n_run, n_train),
            true_fn=lambda: self.pos,
            false_fn=lambda: self._interpolate_pos(n_run, d),
        )
        return x + pos


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int, dropout: float = 0.1, **kwargs):
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


# ---------------------------------------------------------------------------
# SimMIM Decoder
# ---------------------------------------------------------------------------

class PixelReconstructionHead(layers.Layer):
    """SimMIM linear pixel decoder: encoded_patches -> pixel predictions.

    Single Dense layer sufficient — Xie et al. (2022) Table 4 shows
    linear decoder outperforms MLP decoder on reconstruction quality.

    Input  : encoded_patches  (B, N, embed_dim)      from VisionTransformerBackbone
    Output : pixel_pred       (B, N, patch_size^2*C)  pixel-space predictions

    Args:
        patch_size  : ViT patch size matching encoder. Default 16.
        in_channels : Image channels. Default 3.
    """

    def __init__(self, patch_size: int = 16, in_channels: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.patch_size  = patch_size
        self.in_channels = in_channels
        self.out_dim     = patch_size * patch_size * in_channels  # 16^2 * 3 = 768
        self.proj        = layers.Dense(self.out_dim, name='mim_pixel_proj')

    def call(self, encoded_patches: tf.Tensor, training: bool = False) -> tf.Tensor:
        """(B, N, embed_dim) -> (B, N, patch_size^2 * C)"""
        return self.proj(encoded_patches, training=training)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'patch_size': self.patch_size, 'in_channels': self.in_channels})
        return cfg


# ---------------------------------------------------------------------------
# Backbone Models
# ---------------------------------------------------------------------------

class VisionTransformerBackbone(keras.Model):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 8,
        num_heads: int = 6,
        mlp_dim: int = 768,
        dropout: float = 0.1,
        name: str = 'vit_backbone',
    ):
        super().__init__(name=name)
        self.input_spec = keras.layers.InputSpec(shape=(None, *input_shape))
        self.patch_extract = PatchExtract(patch_size)
        self.patch_proj = layers.Dense(embed_dim)
        self.cls_token_layer = LearnableCLSToken()
        self.pos_embed = TrainablePositionalEmbedding()
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f'block_{i}')
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False, return_attention: bool = True) -> dict[str, Any]:
        x = self.patch_extract(x)
        x = self.patch_proj(x)
        x = self.cls_token_layer(x)
        x = self.pos_embed(x)
        attn_all = []
        for i, block in enumerate(self.blocks):
            ret_attn = return_attention and (i == len(self.blocks) - 1)
            if ret_attn:
                x, attn = block(x, training=training, return_attention=True)
                attn_all.append(attn)
            else:
                x = block(x, training=training, return_attention=False)
        x = self.norm(x)
        cls_token = x[:, 0, :]
        encoded_patches = x[:, 1:, :]
        return {
            'cls_token': cls_token,
            'encoded_patches': encoded_patches,
            'last_encoder_layer_attentional_weights': attn_all[-1] if attn_all else None,
            'attention_weights': attn_all,
            'embedding': cls_token,
        }


class SwinLikeBackbone(keras.Model):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (224, 224, 3),
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        name: str = 'swin_like_backbone',
    ):
        super().__init__(name=name)
        self.proj = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')
        self.flatten = layers.Reshape((-1, embed_dim))
        self.pos_embed = TrainablePositionalEmbedding()
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f'swin_block_{i}')
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False, return_attention: bool = True) -> dict[str, Any]:
        fmap = self.proj(x)
        b = tf.shape(fmap)[0]
        h = tf.shape(fmap)[1]
        w = tf.shape(fmap)[2]
        c = tf.shape(fmap)[3]
        x = self.flatten(fmap)
        x = self.pos_embed(x)
        attn_all = []
        for i, block in enumerate(self.blocks):
            ret_attn = return_attention and (i == len(self.blocks) - 1)
            if ret_attn:
                x, attn = block(x, training=training, return_attention=True)
                attn_all.append(attn)
            else:
                x = block(x, training=training, return_attention=False)
        x = self.norm(x)
        gap_vector = tf.reduce_mean(x, axis=1)
        encoded_patches = x
        fmap_out = tf.reshape(encoded_patches, [b, h, w, c])
        return {
            'cls_token': gap_vector,
            'encoded_patches': encoded_patches,
            'last_encoder_layer_attentional_weights': attn_all[-1] if attn_all else None,
            'attention_weights': attn_all,
            'gap_vector': gap_vector,
            'feature_map': fmap_out,
            'embedding': gap_vector,
        }


class CNNBackbone(keras.Model):
    def __init__(self, base_model: keras.Model, name: str = 'cnn_backbone'):
        super().__init__(name=name)
        self.base_model = base_model
        self.gap = layers.GlobalAveragePooling2D(name='gap')

    def call(self, x: tf.Tensor, training: bool = False) -> dict[str, Any]:
        fmap = self.base_model(x, training=training)
        gap = self.gap(fmap)
        return {
            'gap_vector': gap,
            'feature_map': fmap,
            'embedding': gap,
        }


def _make_convnext(input_shape: tuple[int, int, int]) -> keras.Model:
    return keras.applications.ConvNeXtTiny(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None,
    )


def _make_efficientnet(input_shape: tuple[int, int, int]) -> keras.Model:
    return keras.applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=input_shape,
        pooling=None,
    )


class ClassifierWithEncoder(keras.Model):
    def __init__(self, encoder: keras.Model, num_classes: int = 2, dropout: float = 0.2, name: str = 'classifier'):
        super().__init__(name=name)
        self.encoder = encoder
        self.dropout = layers.Dropout(dropout)
        self.head = layers.Dense(num_classes, activation=None, name='classifier_head')

    def call(self, x: tf.Tensor, training: bool = False, return_encoder_outputs: bool = True) -> dict[str, Any]:
        outputs = self.encoder(x, training=training)
        embedding = outputs['embedding']
        logits = self.head(self.dropout(embedding, training=training))
        probs = tf.nn.softmax(logits, axis=-1)
        if return_encoder_outputs:
            out = dict(outputs)
            out['logits'] = logits
            out['probabilities'] = probs
            return out
        return {'logits': logits, 'probabilities': probs}


def build_encoder(name: str, input_shape: tuple[int, int, int] = (224, 224, 3), embed_dim: int = 384) -> keras.Model:
    name = name.lower()
    if name in {'vit', 'vanilla_vit'}:
        return VisionTransformerBackbone(input_shape=input_shape, embed_dim=embed_dim)
    if name in {'swin', 'swin_transformer'}:
        return SwinLikeBackbone(input_shape=input_shape, embed_dim=min(embed_dim, 192))
    if name in {'convnext', 'convnext_tiny'}:
        return CNNBackbone(_make_convnext(input_shape), name='convnext_backbone')
    if name in {'efficientnet', 'efficientnet_b0'}:
        return CNNBackbone(_make_efficientnet(input_shape), name='efficientnet_backbone')
    raise ValueError(f'Unsupported encoder: {name}')


class Classifier(keras.Model):
    def __init__(
        self,
        encoder_name: str,
        input_shape=(224, 224, 3),
        num_classes: int = 2,
        projection_dim: int = 128,
        classifier_hidden_dim: int = 256,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.encoder = build_encoder(encoder_name, input_shape=input_shape)
        self.projection_head = keras.Sequential([
            layers.Dense(classifier_hidden_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(projection_dim),
        ], name='stage2_projection_head')
        self.classifier_head = keras.Sequential([
            layers.Dense(classifier_hidden_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes),
        ], name='stage2_classifier_head')

    def call(self, x, training=False):
        enc = self.encoder(x, training=training)
        base_embedding = enc['embedding']
        projection = self.projection_head(base_embedding, training=training)
        logits = self.classifier_head(base_embedding, training=training)
        return {
            'embedding': base_embedding,
            'projection': projection,
            'logits': logits,
            'probabilities': tf.nn.softmax(logits, axis=-1),
            'tokens': enc.get('tokens', None),
            'features': enc.get('features', None),
        }


def build_classifier(
    encoder_name: str,
    input_shape=(224, 224, 3),
    num_classes: int = 2,
    projection_dim: int = 128,
    classifier_hidden_dim: int = 256,
    dropout_rate: float = 0.2,
):
    return Classifier(
        encoder_name=encoder_name,
        input_shape=input_shape,
        num_classes=num_classes,
        projection_dim=projection_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout_rate=dropout_rate,
    )
