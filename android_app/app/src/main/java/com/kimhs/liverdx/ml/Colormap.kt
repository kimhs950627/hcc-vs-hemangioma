package com.kimhs.liverdx.ml

import kotlin.math.roundToInt

/**
 * Two perceptual ramps so the attention map and the prototype map can never be confused.
 *
 * Both are pure Kotlin (no Android types) and unit-tested: every anchor round-trips exactly and
 * every sample stays inside the alpha-255 sRGB range.
 */
object Colormap {

    /** matplotlib `inferno` sampled at nine points. Attention rollout ramp. */
    val INFERNO = intArrayOf(
        0xFF000004.toInt(), 0xFF1B0C41.toInt(), 0xFF4A0C6B.toInt(),
        0xFF781C6D.toInt(), 0xFFA52C60.toInt(), 0xFFCF4446.toInt(),
        0xFFED6925.toInt(), 0xFFFB9A06.toInt(), 0xFFFCFFA4.toInt(),
    )

    /** Google `turbo` sampled at fifteen points. Prototype-similarity ramp. */
    val TURBO = intArrayOf(
        0xFF30123B.toInt(), 0xFF4145AB.toInt(), 0xFF4675ED.toInt(),
        0xFF39A2FC.toInt(), 0xFF1BCFD4.toInt(), 0xFF24ECA6.toInt(),
        0xFF61FC6C.toInt(), 0xFFA4FC3B.toInt(), 0xFFD1E834.toInt(),
        0xFFF9BA38.toInt(), 0xFFFE9B2D.toInt(), 0xFFF36315.toInt(),
        0xFFD93806.toInt(), 0xFFB11901.toInt(), 0xFF7A0402.toInt(),
    )

    /**
     * Samples a ramp at [t] in [0, 1] with linear interpolation between anchor colours.
     * Values outside [0, 1] clamp to the ends.
     */
    fun sample(anchors: IntArray, t: Float): Int {
        require(anchors.size >= 2) { "a ramp needs at least two anchors" }
        val clamped = t.coerceIn(0f, 1f)
        val position = clamped * (anchors.size - 1)
        val lower = position.toInt().coerceAtMost(anchors.size - 2)
        val fraction = position - lower
        return lerpArgb(anchors[lower], anchors[lower + 1], fraction)
    }

    /** Samples [INFERNO] for the attention rollout map. */
    fun inferno(t: Float): Int = sample(INFERNO, t)

    /** Samples [TURBO] for the HCC prototype-similarity map. */
    fun turbo(t: Float): Int = sample(TURBO, t)

    /** Per-channel linear blend between two opaque sRGB colours, rounded to the nearest step. */
    fun lerpArgb(from: Int, to: Int, fraction: Float): Int {
        val f = fraction.coerceIn(0f, 1f)
        val red = channel(from, 16) + (channel(to, 16) - channel(from, 16)) * f
        val green = channel(from, 8) + (channel(to, 8) - channel(from, 8)) * f
        val blue = channel(from, 0) + (channel(to, 0) - channel(from, 0)) * f
        return (0xFF shl 24) or
            (red.roundToInt() shl 16) or
            (green.roundToInt() shl 8) or
            blue.roundToInt()
    }

    private fun channel(argb: Int, shift: Int): Int = (argb shr shift) and 0xFF
}
