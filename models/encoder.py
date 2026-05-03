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
    N_train = h_train * w_train + 1 (CLS included), determined by
    h_train / w_train passed at construction time (NOT deferred to first call).

    At call time, if the runtime sequence length N_run differs from
    N_train, the patch embeddings are bilinearly interpolated:
        pos  : (1, N_train, D)  ->  reshape to (1, h_t, w_t, D)
               bilinear resize  ->  (1, h_r, w_r, D)
               reshape          ->  (1, N_run, D)

    *** h_train / w_train are REQUIRED. Passing None raises ValueError. ***
    This prevents the pos_embed from being built at the wrong resolution
    when a local-crop view is the first tensor to flow through the model.

    References:
        Dosovitskiy et al. (2020) ViT -- positional embedding interpolation
        Caron et al. (2021) DINO -- local-crop resolution independence
    """

    def __init__(self, h_train: int, w_train: int, **kwargs):
        if h_train is None or w_train is None:
            raise ValueError(
                "TrainablePositionalEmbedding requires h_train and w_train to be "
                "set explicitly at construction time. "
                "Pass h_train=H//patch_size, w_train=W//patch_size from the backbone."
            )
        super().__init__(**kwargs)
        self._h_train = int(h_train)
        self._w_train = int(w_train)
        self._n_train = self._h_train * self._w_train + 1  # +1 for CLS

    def build(self, input_shape):
        d = int(input_shape[-1])
        # Always build with the training-resolution shape, IGNORING input_shape[1].
        # This is the key fix: pos_embed size is determined by h_train/w_train,
        # not by the first tensor that flows through (which could be a local crop).
        n = self._n_train
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
            d     : embedding dim (static int, not tensor)
        Returns:
            Tensor of shape (1, n_run, d)
        """
        cls_pos   = self.pos[:, :1, :]   # (1, 1, D)
        patch_pos = self.pos[:, 1:, :]   # (1, h_train*w_train, D)

        # Use STATIC h_train / w_train -- guaranteed correct by __init__
        h_t = self._h_train
        w_t = self._w_train

        n_patches_run = n_run - 1
        h_r = tf.cast(
            tf.math.round(tf.sqrt(tf.cast(n_patches_run, tf.float32))), tf.int32
        )
        w_r = tf.cast(
            tf.math.ceil(
                tf.cast(n_patches_run, tf.float32) / tf.cast(h_r, tf.float32)
            ),
            tf.int32,
        )

        # Reshape to 2-D spatial grid using STATIC dims -> no shape mismatch
        patch_pos_2d = tf.reshape(patch_pos, [1, h_t, w_t, d])
        patch_pos_2d = tf.image.resize(patch_pos_2d, [h_r, w_r], method='bilinear')
        patch_pos_1d = tf.reshape(patch_pos_2d, [1, h_r * w_r, d])

        return tf.concat([cls_pos, patch_pos_1d], axis=1)  # (1, n_run, D)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        n_run   = tf.shape(x)[1]   # runtime length (int32 tensor)
        d       = x.shape[-1]      # static dim for reshape in interpolation
        n_train = self._n_train    # static Python int

        pos = tf.cond(
            tf.equal(n_run, n_train),
            true_fn=lambda: self.pos,
            false_fn=lambda: self._interpolate_pos(n_run, d),
        )
        return x + pos

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'h_train': self._h_train, 'w_train': self._w_train})
        return cfg


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

    pos_embed is always built at the training resolution (h_train, w_train)
    regardless of which view (global or local) arrives first at runtime.
    Local crops trigger bilinear interpolation in TrainablePositionalEmbedding.
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
        self._patch_size = patch_size

        H, W, _ = input_shape
        if H is None or W is None:
            raise ValueError(
                "VisionTransformerBackbone requires a concrete input_shape (no None dims). "
                f"Got {input_shape}. Pass (224, 224, 3) or similar."
            )
        h_train = H // patch_size
        w_train = W // patch_size

        self.patch_extract     = PatchExtract(patch_size)
        self.patch_proj        = layers.Dense(embed_dim)
        self.cls_token_layer   = LearnableCLSToken()
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
            'last_hidden_state': x,
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
        if H is None or W is None:
            raise ValueError(
                f"SwinLikeBackbone requires concrete input_shape, got {input_shape}"
            )
        h_train = H // patch_size
        w_train = W // patch_size

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

    input_shape MUST be a concrete tuple -- (None, None, 3) is rejected for ViT/Swin
    because TrainablePositionalEmbedding needs h_train / w_train at construction time.
    CNN backbones (convnext, efficientnet) still require a concrete shape for
    Keras applications.
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
