
from __future__ import annotations

from typing import Any

import tensorflow as tf
import keras
from keras import layers

from models.conv_hybrid_vit import ConvHybridViTBackbone


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
    def build(self, input_shape):
        n = int(input_shape[1])
        d = int(input_shape[2])
        self.pos = self.add_weight(
            shape=(1, n, d), initializer='random_normal', trainable=True, name='pos_embed'
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return x + self.pos


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
) -> keras.Model:
    name = name.lower()
    if name in {'vit', 'vanilla_vit'}:
        resolved_patch_size = 16 if patch_size is None else int(patch_size)
        resolved_depth = 8 if depth is None else int(depth)
        resolved_num_heads = 6 if num_heads is None else int(num_heads)
        resolved_mlp_dim = embed_dim * 2 if mlp_dim is None else int(mlp_dim)
        return VisionTransformerBackbone(
            input_shape=input_shape,
            patch_size=resolved_patch_size,
            embed_dim=embed_dim,
            depth=resolved_depth,
            num_heads=resolved_num_heads,
            mlp_dim=resolved_mlp_dim,
            dropout=dropout,
        )
    if name in {'conv_hybrid_vit', 'hybrid_vit', 'convhybridvit'}:
        resolved_patch_size = 16 if patch_size is None else int(patch_size)
        resolved_depth = 8 if depth is None else int(depth)
        resolved_num_heads = 6 if num_heads is None else int(num_heads)
        resolved_mlp_dim = embed_dim * 2 if mlp_dim is None else int(mlp_dim)
        base_grid = (14, 14)
        if input_shape[0] is not None and input_shape[1] is not None:
            base_grid = (max(1, input_shape[0] // resolved_patch_size), max(1, input_shape[1] // resolved_patch_size))
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
