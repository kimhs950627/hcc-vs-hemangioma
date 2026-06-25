"""conv_hybrid_vit.py

EfficientNetV2S → Conv1x1 projection → CLS token prepend → PE → Shallow Transformer

Shape contract (512×512 input, stride-32 backbone)
---------------------------------------------------
  Input       : [B, 512, 512, 3]
  CNN fmap    : [B, 16, 16, 2048]   (ResNet50V2 output stride=32)
  Proj fmap   : [B, 16, 16, d_model]
  Patches     : [B, 256, d_model]   flatten
  + CLS       : [B, 257, d_model]   prepend CLS first
  + PE        : [B, 257, d_model]   PE applied to full sequence (CLS + patches)
  Transformer : n_layers × TransformerEncoderBlock  (sa mode)
             OR n_layers × CrossAttentionEncoderBlock (ca mode)
  Output dict:
    'embedding'          : [B, d_model]          <- CLS token (VICReg loss 용)
    'patch_tokens'       : [B, n_patches, d_model]
    'encoded_patches'    : [B, seq_len, d_model]  <- CLS + patches (legacy key)
    'attention_weights'  : list[ [B, n_heads, seq_len, seq_len] ] × n_layers  (sa)
                        OR list[ [B, n_heads, 1, N_patches]     ] × n_layers  (ca)
    'feature_map'        : [B, gh, gw, d_model]   <- projected fmap (vis 용)
    'patch_grid_size'    : [gh, gw]               <- e.g. [16, 16]

PE mode (cfg.pe_mode):
    'learnable'   : learnable absolute PE [1, 1+n_patches, d_model] — CLS 포함
    'sinusoidal'  : fixed 2-D sinusoidal PE (default) — not updated by optimizer
    False / None : PE 없음 — content-only attention

token_attention_mode (cfg.token_attention_mode):
    'sa'  : Self-Attention — CLS prepend → full sequence SA (default, existing behaviour)
    'ca'  : Cross-Attention — Q = CLS (GAP-initialised per sample), K/V = encoded_patches
            attention_weights shape: [B, n_heads, 1, N_patches]
"""
from __future__ import annotations

import math
from typing import Any, Literal

import keras
import numpy as np
import tensorflow as tf
from keras import layers


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_2d_sinusoidal_pe(seq_len: int, d_model: int) -> np.ndarray:
    """Build 2-D sinusoidal PE for a sequence of length seq_len (CLS + patches).

    Index 0  → CLS : all zeros (no spatial info for CLS)
    Index 1..seq_len-1 → sqrt(seq_len-1) × sqrt(seq_len-1) 2-D grid

    Returns
    -------
    pe : np.ndarray  shape [1, seq_len, d_model]
    """
    n_patches = seq_len - 1
    gh = gw = int(round(math.sqrt(n_patches)))
    assert gh * gw == n_patches, (
        f"n_patches={n_patches} must be a perfect square for 2D sinusoidal PE"
    )
    d_half = d_model // 2          # half for row, half for col
    d_row  = d_half
    d_col  = d_model - d_half

    def _1d_sinusoidal(pos: np.ndarray, dim: int) -> np.ndarray:
        """pos: [N], returns [N, dim]"""
        i = np.arange(dim // 2, dtype=np.float32)
        denom = np.power(10000.0, 2.0 * i / dim)
        angles = pos[:, None] / denom[None, :]          # [N, dim//2]
        pe = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)  # [N, dim]
        if dim % 2 == 1:
            pe = np.concatenate([pe, np.zeros((len(pos), 1), dtype=np.float32)], axis=-1)
        return pe.astype(np.float32)

    rows = np.arange(gh, dtype=np.float32)
    cols = np.arange(gw, dtype=np.float32)

    row_pe = _1d_sinusoidal(rows, d_row)   # [gh, d_row]
    col_pe = _1d_sinusoidal(cols, d_col)   # [gw, d_col]

    # broadcast to 2D grid [gh, gw, d_model]
    grid_pe = np.concatenate(
        [
            np.broadcast_to(row_pe[:, None, :], (gh, gw, d_row)),
            np.broadcast_to(col_pe[None, :, :], (gh, gw, d_col)),
        ],
        axis=-1,
    )  # [gh, gw, d_model]

    patch_pe = grid_pe.reshape(n_patches, d_model)  # [N, d_model]

    # CLS slot: zero vector
    cls_pe = np.zeros((1, d_model), dtype=np.float32)

    pe = np.concatenate([cls_pe, patch_pe], axis=0)[None]  # [1, 1+N, d_model]
    return pe.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# 1-A. TransformerEncoderBlock  (SA mode — unchanged)
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class TransformerEncoderBlock(layers.Layer):
    """Pre-LN Transformer block.  Always collects attention weights.

    attention_weights shape: [B, n_heads, seq_len, seq_len]
    Used in SA mode.
    """

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
# 1-B. CrossAttentionEncoderBlock  (CA mode — NEW)
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class CrossAttentionEncoderBlock(layers.Layer):
    """Pre-LN Cross-Attention block.

    Architecture
    ------------
    Given sequence x [B, 1+N, d_model] (CLS prepended):
      CLS token  : query  [B, 1, d_model]   ← x[:, 0:1, :]
      Patches    : key/value [B, N, d_model] ← x[:, 1:, :]

    Cross-attention:
      Q = LN(cls_token)
      K = V = LN(patch_tokens)
      → MHA(Q, K, V)  → attn_scores [B, n_heads, 1, N_patches]

    CLS update (residual):
      cls_out = cls_token + drop(attn_out)
      cls_out = cls_out + FFN(LN(cls_out))

    Patch tokens pass through unchanged (no SA on patches).
    Output x_out = concat([cls_out, patches], axis=1)  [B, 1+N, d_model]

    attention_weights shape: [B, n_heads, 1, N_patches]
    """

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

        self.norm_q  = layers.LayerNormalization(epsilon=1e-6, name="norm_q")
        self.norm_kv = layers.LayerNormalization(epsilon=1e-6, name="norm_kv")
        self.cross_attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(1, embed_dim // num_heads),
            dropout=dropout,
            name="cross_mha",
        )
        self.drop1 = layers.Dropout(dropout, name="drop_attn")
        self.norm2 = layers.LayerNormalization(epsilon=1e-6, name="norm_ffn")
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
        x: tf.Tensor,            # [B, 1+N, d_model]
        training: bool = False,
        return_attention: bool = True,
    ):
        cls_token = x[:, 0:1, :]    # [B, 1, d_model]
        patches   = x[:, 1:,  :]    # [B, N, d_model]

        q  = self.norm_q(cls_token)   # [B, 1, d_model]
        kv = self.norm_kv(patches)    # [B, N, d_model]

        # MHA: Q=[B,1,d], K=V=[B,N,d] → out=[B,1,d], scores=[B,H,1,N]
        attn_out, attn_scores = self.cross_attn(
            query=q,
            value=kv,
            key=kv,
            return_attention_scores=True,
            training=training,
        )
        cls_updated = cls_token + self.drop1(attn_out, training=training)
        cls_updated = cls_updated + self.mlp(self.norm2(cls_updated), training=training)

        # patches unchanged; reassemble full sequence
        x_out = tf.concat([cls_updated, patches], axis=1)  # [B, 1+N, d_model]

        if return_attention:
            return x_out, attn_scores  # attn_scores: [B, H, 1, N]
        return x_out

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
# 2. ConvPatchEmbedding  — CNN → proj → flatten → CLS prepend → PE → Transformer
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class ConvPatchEmbedding(layers.Layer):
    """CNN-based patch tokenizer with ablation-ready PE.

    Ordering (fixed):
      1. CNN backbone → Conv1x1 proj → LN   [B, gh, gw, d_model]
      2. flatten patches                    [B, N, d_model]
      3. prepend CLS token                  [B, 1+N, d_model]
      4. add PE (mode-dependent)            [B, 1+N, d_model]  ← PE after CLS concat
      5. → TransformerEncoderBlock(s) [SA]  OR  CrossAttentionEncoderBlock(s) [CA]

    pe_mode:
      'learnable'  — learnable abs PE over full seq (CLS + patches); CLS slot init zeros
      'sinusoidal' — fixed 2-D sinusoidal PE; CLS slot = 0; non-trainable (default)
      False/None   — no PE; content-only attention

    token_attention_mode:
      'sa' — self-attention (default); CLS is learnable parameter prepended to patches
      'ca' — cross-attention; CLS token is initialised from GAP of projected patches
             (per-sample dynamic initialisation inside call())
             Learnable CLS weight is NOT used in CA mode (it is initialised but bypassed).
    """

    BACKBONE_MAP = {
        "efficientnetv2_b0": (keras.applications.EfficientNetV2B0, 1280),
        "efficientnetv2_s":  (keras.applications.EfficientNetV2S,  1280),  # ⚠️ stride=8
        "efficientnetv2_m":  (keras.applications.EfficientNetV2M,  1280),  # ⚠️ stride=8
        "resnet50v2":        (keras.applications.ResNet50V2,        2048),
        "densenet121":       (keras.applications.DenseNet121,       1024),
        "convnext_small":    (keras.applications.ConvNeXtSmall,     768),
        "convnext_tiny":     (keras.applications.ConvNeXtTiny,      768),
    }

    def __init__(
        self,
        backbone_name: str = "resnet50v2",
        d_model: int = 256,
        n_patches: int = 256,
        imagenet_pretrained: bool = True,
        backbone_trainable: bool = False,
        pe_mode: str | bool | None = "sinusoidal",
        token_attention_mode: str = "sa",
        # legacy compat — ignored internally; use pe_mode
        use_sinusoidal_pe: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone_name          = backbone_name
        self.d_model                = int(d_model)
        self.n_patches              = int(n_patches)
        self.imagenet_pretrained    = bool(imagenet_pretrained)
        self.backbone_trainable     = bool(backbone_trainable)
        self.token_attention_mode   = str(token_attention_mode).lower()   # 'sa' | 'ca'
        # normalise pe_mode
        if pe_mode is None or pe_mode is False or pe_mode == "none":
            self.pe_mode: str | bool = False
        else:
            self.pe_mode = str(pe_mode)  # "learnable" | "sinusoidal"
        self.use_sinusoidal_pe = use_sinusoidal_pe  # kept for get_config compat only

        if backbone_name not in self.BACKBONE_MAP:
            raise ValueError(f"Unknown backbone: {backbone_name!r}. "
                             f"Choose from {list(self.BACKBONE_MAP)}")
        if backbone_name in ("efficientnetv2_s", "efficientnetv2_m"):
            import warnings
            warnings.warn(
                f"[ConvPatchEmbedding] '{backbone_name}' has output_stride=8. "
                f"512×512 → 4096 patches → OOM risk. Consider 'efficientnetv2_b0'.",
                ResourceWarning, stacklevel=2,
            )
        BackboneCls, _ = self.BACKBONE_MAP[backbone_name]
        self._cnn = BackboneCls(
            include_top=False,
            weights="imagenet" if imagenet_pretrained else None,
        )
        self._cnn.trainable = backbone_trainable
        self._proj      = layers.Conv2D(d_model, kernel_size=1, use_bias=False, name="conv_proj")
        self._proj_norm = layers.LayerNormalization(epsilon=1e-6)

    def build(self, input_shape):
        seq_len = 1 + self.n_patches  # CLS + patches

        # CLS token  [1, 1, d_model]  — always built; bypassed in CA mode at call time
        self.cls_token = self.add_weight(
            shape=(1, 1, self.d_model),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=(self.token_attention_mode == "sa"),  # frozen in CA mode
            name="cls_token",
        )

        # PE weights — applied AFTER CLS concat, over full seq [1, 1+N, d_model]
        if self.pe_mode == "learnable":
            self.pos_embed = self.add_weight(
                shape=(1, seq_len, self.d_model),
                initializer=keras.initializers.TruncatedNormal(stddev=0.02),
                trainable=True,
                name="pos_embed_learnable",
            )
        elif self.pe_mode == "sinusoidal":
            pe_np = _build_2d_sinusoidal_pe(seq_len, self.d_model)
            self.pos_embed = self.add_weight(
                shape=(1, seq_len, self.d_model),
                initializer=keras.initializers.Constant(pe_np),
                trainable=False,   # ← fixed; not updated by optimizer
                name="pos_embed_sinusoidal",
            )
        else:
            # pe_mode is False — no weight needed
            self.pos_embed = None

        super().build(input_shape)

    def call(
        self, x: tf.Tensor, training: bool = False
    ) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Returns
        -------
        tokens : [B, 1+N, d_model]
        fmap   : [B, gh, gw, d_model]
        gh, gw : scalar tensors
        """
        fmap_raw = self._cnn(x, training=(training and self.backbone_trainable))
        fmap     = self._proj(fmap_raw, training=training)
        fmap     = self._proj_norm(fmap, training=training)

        B  = tf.shape(x)[0]
        gh = tf.shape(fmap)[1]
        gw = tf.shape(fmap)[2]
        N  = gh * gw

        # 1. flatten patches  [B, N, d_model]
        patches = tf.reshape(fmap, [B, N, self.d_model])

        # 2. CLS token initialisation
        if self.token_attention_mode == "ca":
            # CA mode: per-sample GAP of projected patches → [B, 1, d_model]
            cls = tf.reduce_mean(patches, axis=1, keepdims=True)   # [B, 1, d_model]
        else:
            # SA mode: learnable CLS token (tiled across batch)
            cls = tf.cast(tf.tile(self.cls_token, [B, 1, 1]), patches.dtype)

        # 3. prepend CLS  [B, 1+N, d_model]
        tokens = tf.concat([cls, patches], axis=1)

        # 4. add PE over full sequence (CLS + patches)
        if self.pos_embed is not None:
            tokens = tokens + tf.cast(self.pos_embed, tokens.dtype)

        return tokens, fmap, gh, gw

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            backbone_name=self.backbone_name,
            d_model=self.d_model,
            n_patches=self.n_patches,
            imagenet_pretrained=self.imagenet_pretrained,
            backbone_trainable=self.backbone_trainable,
            pe_mode=self.pe_mode if self.pe_mode else False,
            token_attention_mode=self.token_attention_mode,
            use_sinusoidal_pe=self.use_sinusoidal_pe,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 3. ConvHybridViTBackbone  — full encoder
# ──────────────────────────────────────────────────────────────────────────────
@keras.saving.register_keras_serializable(package="hcc")
class ConvHybridViTBackbone(keras.Model):
    """CNN backbone + Shallow Transformer with CLS token.

    PE ablation controlled by cfg.pe_mode:
        'sinusoidal'  (default) — fixed 2-D sinusoidal, CLS slot = 0
        'learnable'             — learnable abs PE over full seq
        False / None             — no PE

    token_attention_mode (cfg.token_attention_mode):
        'sa'  (default) — Self-Attention; full sequence [CLS+patches]; attn [B,H,seq,seq]
        'ca'            — Cross-Attention; Q=CLS(GAP), K/V=patches; attn [B,H,1,N_patches]

    Output dict
    -----------
    'embedding'          : [B, d_model]
    'patch_tokens'       : [B, n_patches, d_model]
    'encoded_patches'    : [B, seq_len, d_model]
    'attention_weights'  : list of [B, n_heads, seq_len, seq_len]  (sa mode)
                        OR list of [B, n_heads, 1, N_patches]       (ca mode)
    'feature_map'        : [B, gh, gw, d_model]
    'patch_grid_size'    : [gh, gw]
    'gap_vector'         : alias for 'embedding'
    'token_attention_mode': str — 'sa' or 'ca' (for downstream branching)
    """

    def __init__(
        self,
        input_shape: tuple = (512, 512, 3),
        backbone_name: str = "resnet50v2",
        d_model: int = 256,
        n_patches: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        imagenet_pretrained: bool = True,
        backbone_trainable: bool = False,
        pe_mode: str | bool | None = "sinusoidal",
        token_attention_mode: str = "sa",
        # legacy compat
        use_sinusoidal_pe: bool = True,
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
        self.pool_mode           = pool_mode
        self._token_attn_mode    = str(token_attention_mode).lower()  # 'sa' | 'ca'
        # normalise pe_mode
        if pe_mode is None or pe_mode is False or pe_mode == "none":
            self.pe_mode: str | bool = False
        else:
            self.pe_mode = str(pe_mode)
        self.use_sinusoidal_pe = use_sinusoidal_pe  # compat

        self.patch_embed = ConvPatchEmbedding(
            backbone_name=backbone_name,
            d_model=d_model,
            n_patches=n_patches,
            imagenet_pretrained=imagenet_pretrained,
            backbone_trainable=backbone_trainable,
            pe_mode=pe_mode,
            token_attention_mode=token_attention_mode,
            use_sinusoidal_pe=use_sinusoidal_pe,
            name="patch_embed",
        )

        # ── Block selection: SA vs CA ────────────────────────────────────────
        if self._token_attn_mode == "ca":
            self.blocks = [
                CrossAttentionEncoderBlock(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    dropout=dropout,
                    name=f"ca_block_{i}",
                )
                for i in range(depth)
            ]
        else:
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

    # ── Property to expose mode externally ────────────────────────────────────
    @property
    def token_attention_mode(self) -> str:
        return self._token_attn_mode

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
            # attn shape:
            #   SA mode: [B, n_heads, seq_len, seq_len]
            #   CA mode: [B, n_heads, 1, N_patches]

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
            "token_attention_mode": self._token_attn_mode,
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
            pe_mode=self.pe_mode if self.pe_mode else False,
            token_attention_mode=self._token_attn_mode,
            use_sinusoidal_pe=self.use_sinusoidal_pe,
            pool_mode=self.pool_mode,
        )
        return cfg


# ──────────────────────────────────────────────────────────────────────────────
# 4. Factory
# ──────────────────────────────────────────────────────────────────────────────
def build_conv_hybrid_vit(
    input_shape: tuple = (512, 512, 3),
    backbone_name: str = "resnet50v2",
    d_model: int = 256,
    n_patches: int = 256,
    depth: int = 2,
    num_heads: int = 8,
    mlp_dim: int = 512,
    dropout: float = 0.1,
    imagenet_pretrained: bool = True,
    backbone_trainable: bool = False,
    pe_mode: str | bool | None = "sinusoidal",
    token_attention_mode: str = "sa",
    # legacy compat
    use_sinusoidal_pe: bool = True,
    pool_mode: str = "cls",
) -> ConvHybridViTBackbone:
    """Build and warm-start ConvHybridViTBackbone.

    pe_mode options
    ---------------
    'sinusoidal'  — fixed 2-D sinusoidal PE (default; recommended for ablation)
    'learnable'   — fully trainable absolute PE over CLS + patch tokens
    False          — no PE (content-only attention baseline)

    token_attention_mode options
    ----------------------------
    'sa'  (default) — Self-Attention over full sequence [CLS + patches]
                      attention_weights: [B, n_heads, seq_len, seq_len]
    'ca'            — Cross-Attention; Q=CLS (GAP-initialised per sample),
                      K/V=encoded_patches
                      attention_weights: [B, n_heads, 1, N_patches]

    Typical notebook usage
    ----------------------
    # In cfg cell:
    #   cfg.token_attention_mode = 'sa'  # or 'ca'
    encoder = build_conv_hybrid_vit(
        ...
        token_attention_mode=cfg.token_attention_mode,
    )
    """
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
        pe_mode=pe_mode,
        token_attention_mode=token_attention_mode,
        use_sinusoidal_pe=use_sinusoidal_pe,
        pool_mode=pool_mode,
    )
    dummy = tf.zeros((2, *input_shape))
    _ = model(dummy, training=False)
    h, w, _ = input_shape
    gh, gw  = h // 32, w // 32
    n_patch = gh * gw
    pe_label = pe_mode if pe_mode else "none"
    print(
        f"[ConvHybridViT] backbone={backbone_name}  "
        f"input={input_shape}  fmap={gh}×{gw}  n_patch={n_patch}  "
        f"seq_len=1+{n_patch}={1+n_patch}  d_model={d_model}  "
        f"depth={depth}  n_heads={num_heads}  PE={pe_label}  "
        f"attn_mode={token_attention_mode.upper()}"
    )
    if token_attention_mode == "ca":
        print(
            f"[ConvHybridViT] CA mode: attn_weights shape per layer = "
            f"[B, {num_heads}, 1, {n_patch}]"
        )
    else:
        print(
            f"[ConvHybridViT] SA mode: attn_weights shape per layer = "
            f"[B, {num_heads}, {1+n_patch}, {1+n_patch}]"
        )
    print(
        f"[ConvHybridViT] total_params={model.count_params():,}  "
        f"trainable={sum(tf.size(v).numpy() for v in model.trainable_variables):,}"
    )
    return model
