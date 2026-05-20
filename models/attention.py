from __future__ import annotations

import tensorflow as tf
import keras


_NEG_INF = -1e9


@keras.saving.register_keras_serializable(package="hcc")
class AttentionDropMultiHeadAttention(keras.layers.MultiHeadAttention):
    """Official Keras MHA + optional AttentionDrop.

    - attn_drop=False: upstream MultiHeadAttention path
    - attn_drop=True and training=True: apply extra top-k Bernoulli masking
      on attention logits before softmax.
    """

    def __init__(
        self,
        *args,
        attn_drop: bool = False,
        attn_drop_rate: float = 0.0,
        attn_drop_top_k: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.attn_drop = bool(attn_drop)
        self.attn_drop_rate = float(attn_drop_rate)
        self.attn_drop_top_k = int(attn_drop_top_k)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                'attn_drop': self.attn_drop,
                'attn_drop_rate': self.attn_drop_rate,
                'attn_drop_top_k': self.attn_drop_top_k,
            }
        )
        return cfg

    def _attention_drop_additive_mask(self, scores: tf.Tensor) -> tf.Tensor:
        shape = tf.shape(scores)
        b, n, t, s = shape[0], shape[1], shape[2], shape[3]

        static_s = scores.shape[-1]
        if static_s is None:
            k = tf.minimum(tf.cast(self.attn_drop_top_k, tf.int32), s)
        else:
            k = min(self.attn_drop_top_k, int(static_s))

        _, top_idx = tf.math.top_k(scores, k=k)
        drop_flag = tf.cast(
            tf.random.uniform(tf.shape(top_idx), dtype=scores.dtype)
            < tf.cast(self.attn_drop_rate, scores.dtype),
            scores.dtype,
        )

        flat_scores = tf.reshape(scores, [-1, s])
        flat_idx = tf.reshape(top_idx, [-1, k])
        flat_drop = tf.reshape(drop_flag, [-1, k])
        num_rows = tf.shape(flat_scores)[0]

        row_idx = tf.repeat(tf.expand_dims(tf.range(num_rows), 1), repeats=k, axis=1)
        scatter_idx = tf.stack(
            [tf.reshape(row_idx, [-1]), tf.reshape(flat_idx, [-1])],
            axis=1,
        )
        scatter_vals = tf.reshape(flat_drop, [-1]) * tf.cast(_NEG_INF, scores.dtype)

        flat_mask = tf.tensor_scatter_nd_update(
            tf.zeros_like(flat_scores),
            scatter_idx,
            scatter_vals,
        )
        return tf.reshape(flat_mask, [b, n, t, s])

    def _compute_attention(
        self,
        query,
        key,
        value,
        attention_mask=None,
        training=None,
        return_attention_scores=False,
    ):
        if (not self.attn_drop) or (not training) or self.attn_drop_rate <= 0.0:
            return super()._compute_attention(
                query=query,
                key=key,
                value=value,
                attention_mask=attention_mask,
                training=training,
                return_attention_scores=return_attention_scores,
            )

        query = tf.multiply(query, tf.cast(self._inverse_sqrt_key_dim, query.dtype))
        scores = tf.einsum(self._dot_product_equation, key, query)

        if attention_mask is not None:
            mask = tf.cast(attention_mask, scores.dtype)
            mask_min = tf.reduce_min(mask)
            mask_max = tf.reduce_max(mask)

            def _bool_mask():
                return scores + (1.0 - mask) * tf.cast(_NEG_INF, scores.dtype)

            def _additive_mask():
                return scores + mask

            scores = tf.cond(
                tf.logical_and(mask_min >= 0.0, mask_max <= 1.0),
                _bool_mask,
                _additive_mask,
            )

        scores = scores + self._attention_drop_additive_mask(scores)
        probs = self._masked_softmax(scores, attention_mask=None)
        probs = self._dropout_layer(probs, training=training)
        attention_output = tf.einsum(self._combine_equation, probs, value)

        if return_attention_scores:
            return attention_output, probs
        return attention_output, None
