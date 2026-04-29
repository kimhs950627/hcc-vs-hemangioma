from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras
from keras import layers


class PatchExtract(layers.Layer):
    """Non-overlapping patch extraction via tf.image.extract_patches."""

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
    """Prepend a trainable [CLS] token to the patch sequence."""

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
    """Learnable 1-D positional embedding with dynamic interpolation.

    At build time, the weight shape is fixed to (1, N_train, D) where
    N_train is the sequence length seen during the first call (or build).

    At call time, if the runtime sequence length N_run differs from
    N_train, the embedding is bilinearly interpolated:
        pos  : (1, N_train, D)  ->  reshape to (1, h_t, w_t, D)
               bilinear resize  ->  (1, h_r, w_r, D)
               reshape          ->  (1, N_run, D)

    This is exactly the approach used in original ViT and DINOv1/v2
    for handling arbitrary resolution at inference / local-crop training.

    Assumption: the sequence represents a 2-D grid of patches, so
    sqrt(N_train) and sqrt(N_run) must be integers (square grids).
    For non-square grids, pass h_train / w_train at build time.
    """

    def __init__(self, h_train: int | None = None, w_train: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._h_train = h_train
        self._w_train = w_train

    def build(self, input_shape):
        n = int(input_shape[1])  # sequence length at build time
        d = int(input_shape[2])  # embedding dim
        # Infer (h, w) of the patch grid
        if self._h_train is not None and self._w_train is not None:
            h, w = self._h_train, self._w_train
        else:
            # Assumes square grid; CLS token is already prepended so N=N_patches+1
            # We store ALL positions including CLS (position 0).
            h = w = int((n - 1) ** 0.5) if n > 1 else 1
        self._n_train = n
        self._h_built  = h
        self._w_built  = w
        self.pos = self.add_weight(
            shape=(1, n, d),
            initializer='random_normal',
            trainable=True,
            name='pos_embed',
        )
        super().build(input_shape)

    def _interpolate_pos(self, n_run: int, d: int) -> tf.Tensor:
        """Bilinearly interpolate pos_embed from N_train to N_run.

        Args:
            n_run : runtime sequence length (including CLS token)
            d     : embedding dim
        Returns:
            Tensor of shape (1, n_run, d)
        """
        # Split CLS and patch positions
        cls_pos   = self.pos[:, :1, :]                  # (1, 1, D)
        patch_pos = self.pos[:, 1:, :]                   # (1, N_patches_train, D)

        h_t = self._h_built
        w_t = self._w_built

        # N_patches_run (excluding CLS)
        n_patches_run = n_run - 1
        h_r = tf.cast(tf.math.round(tf.sqrt(tf.cast(n_patches_run, tf.float32))), tf.int32)
        w_r = tf.cast(tf.math.ceil(tf.cast(n_patches_run, tf.float32) / tf.cast(h_r, tf.float32)), tf.int32)

        # (1, h_t*w_t, D) -> (1, h_t, w_t, D) -> bilinear -> (1, h_r, w_r, D) -> (1, h_r*w_r, D)
        patch_pos_2d = tf.reshape(patch_pos, [1, h_t, w_t, d])
        patch_pos_2d = tf.image.resize(patch_pos_2d, [h_r, w_r], method='bilinear')
        patch_pos_1d = tf.reshape(patch_pos_2d, [1, h_r * w_r, d])

        return tf.concat([cls_pos, patch_pos_1d], axis=1)  # (1, n_run, D)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        n_run = tf.shape(x)[1]    # runtime sequence length (int32 tensor)
        d     = x.shape[-1]       # static dim (needed for reshape in interpolation)

        # Fast path: runtime length == trained length
        n_train = self._n_train
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
        self.attn  = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
        )
        self.drop1 = layers.Dropout(dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp   = keras.Sequential([
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
            attn_out   = self.attn(y, y, training=training)
            attn_scores = None
        x = x + self.drop1(attn_out, training=training)
        x = x + self.mlp(self.norm2(x), training=training)
        if return_attention:
            return x, attn_scores
        return x


class VisionTransformerBackbone(keras.Model):
    """Vision Transformer backbone with dynamic positional embedding interpolation.

    input_shape may be (None, None, 3) for fully flexible resolution,
    or a fixed tuple like (224, 224, 3) to pre-infer N_train for pos_embed.

    When local crops (e.g. 96 x 96) are passed at training time, the
    TrainablePositionalEmbedding layer automatically interpolates the
    stored pos_embed to match the runtime patch-grid size.
    No upscaling of the input image is needed.
    """

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
        # NOTE: input_spec intentionally removed so model accepts arbitrary (H, W).
        self._patch_size = patch_size

        # Pre-compute training-resolution patch grid for pos_embed initialisation
        H, W, _ = input_shape
        if H is not None and W is not None:
            h_train = H // patch_size
            w_train = W // patch_size
        else:
            h_train = w_train = None   # resolved lazily on first forward pass

        self.patch_extract     = PatchExtract(patch_size)
        self.patch_proj        = layers.Dense(embed_dim)
        self.cls_token_layer   = LearnableCLSToken()
        # Pass h/w so pos_embed knows the trained grid shape for interpolation.
        self.pos_embed         = TrainablePositionalEmbedding(
            h_train=h_train, w_train=w_train, name='pos_embed'
        )
        self.blocks = [
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout, name=f'block_{i}')
            for i in range(depth)
        ]
        self.norm = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x: tf.Tensor, training: bool = False, return_attention: bool = True) -> dict[str, Any]:
        x = self.patch_extract(x)        # (B, N_patches, patch_dim)
        x = self.patch_proj(x)           # (B, N_patches, D)
        x = self.cls_token_layer(x)      # (B, N_patches+1, D)
        x = self.pos_embed(x)            # (B, N_patches+1, D) -- interpolated if N differs
        attn_all = []
        for i, block in enumerate(self.blocks):
            ret_attn = return_attention and (i == len(self.blocks) - 1)
            if ret_attn:
                x, attn = block(x, training=training, return_attention=True)
                attn_all.append(attn)
            else:
                x = block(x, training=training, return_attention=False)
        x = self.norm(x)
        cls_token       = x[:, 0, :]
        encoded_patches = x[:, 1:, :]
        return {
            'cls_token':    cls_token,
            'encoded_patches': encoded_patches,
            'last_hidden_state': x,        # (B, N+1, D)  --  used by SimMIM head
            'last_encoder_layer_attentional_weights': attn_all[-1] if attn_all else None,
            'attention_weights': attn_all,
            'embedding':    cls_token,
        }


class SwinLikeBackbone(keras.Model):
    """Swin-like backbone with dynamic pos_embed interpolation."""

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
        H, W, _ = input_shape
        if H is not None and W is not None:
            h_train = H // patch_size
            w_train = W // patch_size
        else:
            h_train = w_train = None

        self.proj    = layers.Conv2D(embed_dim, kernel_size=patch_size, strides=patch_size, padding='valid')
        self.flatten = layers.Reshape((-1, embed_dim))
        self.pos_embed = TrainablePositionalEmbedding(
            h_train=h_train, w_train=w_train, name='pos_embed'
        )
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
        gap_vector      = tf.reduce_mean(x, axis=1)
        encoded_patches = x
        fmap_out        = tf.reshape(encoded_patches, [b, h, w, c])
        return {
            'cls_token':    gap_vector,
            'encoded_patches': encoded_patches,
            'last_encoder_layer_attentional_weights': attn_all[-1] if attn_all else None,
            'attention_weights': attn_all,
            'gap_vector':   gap_vector,
            'feature_map':  fmap_out,
            'embedding':    gap_vector,
        }


class CNNBackbone(keras.Model):
    def __init__(self, base_model: keras.Model, name: str = 'cnn_backbone'):
        super().__init__(name=name)
        self.base_model = base_model
        self.gap        = layers.GlobalAveragePooling2D(name='gap')

    def call(self, x: tf.Tensor, training: bool = False) -> dict[str, Any]:
        fmap = self.base_model(x, training=training)
        gap  = self.gap(fmap)
        return {
            'gap_vector':  gap,
            'feature_map': fmap,
            'embedding':   gap,
        }


# ---------------------------------------------------------------------------
# Pre-trained CNN factories
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class ClassifierWithEncoder(keras.Model):
    def __init__(self, encoder: keras.Model, num_classes: int = 2, dropout: float = 0.2, name: str = 'classifier'):
        super().__init__(name=name)
        self.encoder = encoder
        self.dropout = layers.Dropout(dropout)
        self.head    = layers.Dense(num_classes, activation=None, name='classifier_head')

    def call(self, x: tf.Tensor, training: bool = False, return_encoder_outputs: bool = True) -> dict[str, Any]:
        outputs   = self.encoder(x, training=training)
        embedding = outputs['embedding']
        logits    = self.head(self.dropout(embedding, training=training))
        probs     = tf.nn.softmax(logits, axis=-1)
        if return_encoder_outputs:
            out = dict(outputs)
            out['logits']        = logits
            out['probabilities'] = probs
            return out
        return {'logits': logits, 'probabilities': probs}


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
        enc           = self.encoder(x, training=training)
        base_embedding = enc['embedding']
        projection    = self.projection_head(base_embedding, training=training)
        logits        = self.classifier_head(base_embedding, training=training)
        return {
            'embedding':     base_embedding,
            'projection':    projection,
            'logits':        logits,
            'probabilities': tf.nn.softmax(logits, axis=-1),
            'tokens':        enc.get('tokens', None),
            'features':      enc.get('features', None),
        }


# ---------------------------------------------------------------------------
# Public factories
# ---------------------------------------------------------------------------

def build_encoder(
    name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    embed_dim: int = 384,
) -> keras.Model:
    """Build an encoder backbone.

    input_shape may be (None, None, 3) for resolution-agnostic ViT.
    For CNN backbones, a concrete shape is still required by Keras applications.
    """
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
