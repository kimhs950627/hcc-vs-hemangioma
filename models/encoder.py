from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras
from keras import layers

from models.conv_hybrid_vit import ConvHybridViTBackbone, OptionalAbsolutePositionalEmbedding
from models.attention import AttentionDropMultiHeadAttention


@keras.saving.register_keras_serializable(package="hcc")
class PatchExtract(layers.Layer):
    def __init__(self, patch_size: int = 16, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        p = self.patch_size
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, p, p, 1],
            strides=[1, p, p, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        gh = tf.shape(patches)[1]
        gw = tf.shape(patches)[2]
        patch_dim = tf.shape(patches)[-1]
        return tf.reshape(patches, [tf.shape(images)[0], -1, patch_dim]), gh, gw


@keras.saving.register_keras_serializable(package="hcc")
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

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="hcc")
class TransformerBlock(layers.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.1,
        attn_drop: bool = False,
        attn_drop_rate: float = 0.0,
        attn_drop_top_k: int = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.mlp_dim = int(mlp_dim)
        self.dropout = float(dropout)
        self.attn_drop = bool(attn_drop)
        self.attn_drop_rate = float(attn_drop_rate)
        self.attn_drop_top_k = int(attn_drop_top_k)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        if attn_drop:
            self.attn = AttentionDropMultiHeadAttention(
                num_heads=num_heads,
                key_dim=max(1, embed_dim // num_heads),
                dropout=dropout,
                attn_drop=True,
                attn_drop_rate=attn_drop_rate,
                attn_drop_top_k=attn_drop_top_k,
            )
        else:
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


@keras.saving.register_keras_serializable(package="hcc")
class VisionTransformerBackbone(keras.Model):
    """Pure ViT backbone with variable-resolution support.

    Patch embedding
    ---------------
    tf.image.extract_patches (no Conv2D) -> Dense projection
    → no spatial inductive bias, uniform treatment of all patch positions

    Positional encoding
    -------------------
    OptionalAbsolutePositionalEmbedding (bicubic interpolation, shared with ConvHybridViT)
    → learnable (1, base_gh, base_gw, D) grid, resized at runtime to actual (gh, gw)
    → supports arbitrary input resolutions without shape errors

    CLS token
    ---------
    prepended after patch projection, before PE
    PE for CLS uses a dedicated learnable scalar (use_cls_token=True)

    Shape contract
    --------------
    Input : (B, H, W, 3), H and W must be multiples of patch_size
    Output: cls_token (B, D), encoded_patches (B, N, D)
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
        use_positional_encoding: bool = True,
        base_grid_size: tuple[int, int] = (14, 14),
        attn_drop: bool = False,
        attn_drop_rate: float = 0.0,
        attn_drop_top_k: int = 2,
        name: str = 'vit_backbone',
    ):
        super().__init__(name=name)
        self.input_spec = keras.layers.InputSpec(ndim=4, axes={-1: input_shape[-1]})
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_positional_encoding = bool(use_positional_encoding)

        self.patch_extract = PatchExtract(patch_size, name='patch_extract')
        self.patch_proj    = layers.Dense(embed_dim, use_bias=True, name='patch_proj')
        self.cls_token_layer = LearnableCLSToken(name='cls_token')

        # Shared PE class with ConvHybridViT — bicubic interpolation at runtime
        # use_cls_token=True: separate learnable scalar for the CLS position
        self.pos_embed = OptionalAbsolutePositionalEmbedding(
            embed_dim=embed_dim,
            base_grid_size=base_grid_size,
            use_cls_token=True,
            name='pos_embed',
        )

        self.blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                mlp_dim,
                dropout,
                attn_drop=attn_drop,
                attn_drop_rate=attn_drop_rate,
                attn_drop_top_k=attn_drop_top_k,
                name=f'block_{i}',
            )
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

        # (B,H,W,3) → extract → (B,N,p*p*3) → Dense → (B,N,D)
        patches, gh, gw = self.patch_extract(x)
        tokens = self.patch_proj(patches)

        # prepend CLS: (B,N,D) → (B,N+1,D)
        tokens = self.cls_token_layer(tokens)

        # bicubic PE (gh, gw for patch grid; CLS gets its own learnable pos)
        if self.use_positional_encoding:
            tokens = self.pos_embed(tokens, grid_hw=(gh, gw))

        attn_all = []
        y = tokens
        for i, block in enumerate(self.blocks):
            ret_attn = bool(return_attention)
            if ret_attn:
                y, attn = block(y, training=training, return_attention=True)
                attn_all.append(attn)
            else:
                y = block(y, training=training, return_attention=False)
        y = self.norm(y)

        cls_token       = y[:, 0, :]
        encoded_patches = y[:, 1:, :]
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
        self.pos_embed = OptionalAbsolutePositionalEmbedding(
            embed_dim=embed_dim,
            base_grid_size=(56, 56),
            use_cls_token=False,
            name='pos_embed',
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
        tokens = self.flatten(fmap)
        tokens = self.pos_embed(tokens, grid_hw=(h, w))
        attn_all = []
        for i, block in enumerate(self.blocks):
            ret_attn = bool(return_attention)
            if ret_attn:
                tokens, attn = block(tokens, training=training, return_attention=True)
                attn_all.append(attn)
            else:
                tokens = block(tokens, training=training, return_attention=False)
        tokens = self.norm(tokens)
        gap_vector = tf.reduce_mean(tokens, axis=1)
        encoded_patches = tokens
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


def build_encoder(
    name: str,
    input_shape: tuple[int, int, int] = (224, 224, 3),
    embed_dim: int = 384,
    depth: int | None = None,
    num_heads: int | None = None,
    mlp_dim: int | None = None,
    patch_size: int | None = None,
    dropout: float = 0.1,
    use_pe: bool = False,
    attn_drop: bool = False,
    attn_drop_rate: float = 0.0,
    attn_drop_top_k: int = 2,
) -> keras.Model:
    name = name.lower()
    if name in {'vit', 'vanilla_vit'}:
        resolved_patch_size = 16 if patch_size is None else int(patch_size)
        resolved_depth      = 8  if depth is None else int(depth)
        resolved_num_heads  = 6  if num_heads is None else int(num_heads)
        resolved_mlp_dim    = embed_dim * 2 if mlp_dim is None else int(mlp_dim)
        base_grid = (14, 14)
        if input_shape[0] is not None and input_shape[1] is not None:
            base_grid = (
                max(1, input_shape[0] // resolved_patch_size),
                max(1, input_shape[1] // resolved_patch_size),
            )
        return VisionTransformerBackbone(
            input_shape=input_shape,
            patch_size=resolved_patch_size,
            embed_dim=embed_dim,
            depth=resolved_depth,
            num_heads=resolved_num_heads,
            mlp_dim=resolved_mlp_dim,
            dropout=dropout,
            use_positional_encoding=use_pe,
            base_grid_size=base_grid,
            attn_drop=attn_drop,
            attn_drop_rate=attn_drop_rate,
            attn_drop_top_k=attn_drop_top_k,
        )
    if name in {'conv_hybrid_vit', 'hybrid_vit', 'convhybridvit'}:
        resolved_patch_size = 16 if patch_size is None else int(patch_size)
        resolved_depth      = 8  if depth is None else int(depth)
        resolved_num_heads  = 6  if num_heads is None else int(num_heads)
        resolved_mlp_dim    = embed_dim * 2 if mlp_dim is None else int(mlp_dim)
        base_grid = (14, 14)
        if input_shape[0] is not None and input_shape[1] is not None:
            base_grid = (
                max(1, input_shape[0] // resolved_patch_size),
                max(1, input_shape[1] // resolved_patch_size),
            )
        return ConvHybridViTBackbone(
            input_shape=(None, None, input_shape[-1]),
            patch_size=resolved_patch_size,
            embed_dim=embed_dim,
            depth=resolved_depth,
            num_heads=resolved_num_heads,
            mlp_dim=resolved_mlp_dim,
            dropout=dropout,
            conv_stem_depth=2,
            stem_kernel_size=3,
            use_positional_encoding=use_pe,
            base_grid_size=base_grid,
            pool_mode='gap',
        )
    if name in {'swin', 'swin_transformer'}:
        return SwinLikeBackbone(input_shape=input_shape, embed_dim=min(embed_dim, 192))
    if name in {'convnext', 'convnext_tiny'}:
        return CNNBackbone(_make_convnext(input_shape), name='convnext_backbone')
    if name in {'efficientnet', 'efficientnet_b0'}:
        return CNNBackbone(_make_efficientnet(input_shape), name='efficientnet_backbone')
    raise ValueError(f'Unsupported encoder: {name}')


def build_classifier(name: str, input_shape: tuple[int, int, int] = (224, 224, 3), num_classes: int = 2, use_pe: bool = False) -> ClassifierWithEncoder:
    encoder = build_encoder(name=name, input_shape=input_shape, use_pe=use_pe)
    return ClassifierWithEncoder(encoder=encoder, num_classes=num_classes, name=f'{name}_classifier')


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
