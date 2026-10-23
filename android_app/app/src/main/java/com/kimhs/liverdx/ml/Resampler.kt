package com.kimhs.liverdx.ml

/**
 * Resamples a decoded frame to the model input size.
 *
 * Two modes exist because the repository contains two different preprocessing paths and they
 * disagree:
 *  - [AREA] mirrors `inference/explainable_predictor.py` (what the demo/visualisation figures use),
 *  - [BILINEAR_UINT8] mirrors `dataloader.py::_decode_image`, i.e. `tf.image.resize(bilinear)` then
 *    `tf.cast(uint8)`, which is the pipeline the reported validation/test metrics came from.
 *
 * Both are fixture-verified against their Python counterparts; the user picks which one the app runs.
 */
enum class ResizeMode(val id: String, val label: String, val source: String) {
    BILINEAR_UINT8("bilinear_uint8", "bilinear + uint8", "dataloader.py · reported metrics"),
    AREA("area", "area average", "explainable_predictor.py · demo pipeline"),
}

object Resampler {

    /** Row-major RGB floats in [0, 255] for a `dstWidth x dstHeight` frame. */
    fun resample(
        rgb: FloatArray,
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
        mode: ResizeMode,
    ): FloatArray = when (mode) {
        ResizeMode.AREA -> areaAverage(rgb, srcWidth, srcHeight, dstWidth, dstHeight)
        ResizeMode.BILINEAR_UINT8 -> bilinearUint8(rgb, srcWidth, srcHeight, dstWidth, dstHeight)
    }

    /**
     * Coverage-weighted (area) downscale, matching `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
     *
     * Canonical implementation lives in [Resampler]; [ScoreMath.resizeAreaAvg] provides the ARGB-int
     * entry point used by the fixture tests.
     */
    fun areaAverage(rgb: FloatArray, srcWidth: Int, srcHeight: Int, dstWidth: Int, dstHeight: Int): FloatArray {
        require(rgb.size >= srcWidth * srcHeight * 3) {
            "RGB buffer holds ${rgb.size} values, need ${srcWidth * srcHeight * 3}"
        }
        val output = FloatArray(dstWidth * dstHeight * 3)
        val scaleX = srcWidth.toDouble() / dstWidth
        val scaleY = srcHeight.toDouble() / dstHeight

        for (dy in 0 until dstHeight) {
            val top = dy * scaleY
            val bottom = top + scaleY
            val rowStart = floorToInt(top).coerceIn(0, srcHeight - 1)
            val rowEnd = ceilToInt(bottom).coerceIn(rowStart + 1, srcHeight)
            for (dx in 0 until dstWidth) {
                val left = dx * scaleX
                val right = left + scaleX
                val colStart = floorToInt(left).coerceIn(0, srcWidth - 1)
                val colEnd = ceilToInt(right).coerceIn(colStart + 1, srcWidth)

                val total = DoubleArray(3)
                var weightSum = 0.0
                for (y in rowStart until rowEnd) {
                    val rowWeight = overlap(y.toDouble(), y + 1.0, top, bottom)
                    if (rowWeight <= 0.0) continue
                    for (x in colStart until colEnd) {
                        val colWeight = overlap(x.toDouble(), x + 1.0, left, right)
                        if (colWeight <= 0.0) continue
                        val weight = rowWeight * colWeight
                        val base = (y * srcWidth + x) * 3
                        for (channel in 0 until 3) total[channel] += rgb[base + channel] * weight
                        weightSum += weight
                    }
                }

                val inverse = if (weightSum > 0.0) 1.0 / weightSum else 0.0
                val base = (dy * dstWidth + dx) * 3
                for (channel in 0 until 3) {
                    output[base + channel] = (total[channel] * inverse).toFloat()
                }
            }
        }
        return output
    }

    private fun overlap(segmentStart: Double, segmentEnd: Double, windowStart: Double, windowEnd: Double): Double {
        val low = maxOf(segmentStart, windowStart)
        val high = minOf(segmentEnd, windowEnd)
        return if (high > low) high - low else 0.0
    }

    private fun floorToInt(value: Double): Int = kotlin.math.floor(value).toInt()

    private fun ceilToInt(value: Double): Int = kotlin.math.ceil(value).toInt()

    /**
     * Plain bilinear with half-pixel centres and no antialias, then truncation to the uint8 domain —
     * exactly `tf.image.resize(..., method="bilinear")` followed by `tf.cast(..., uint8)`.
     */
    fun bilinearUint8(
        pixels: FloatArray,
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
    ): FloatArray {
        val output = FloatArray(dstWidth * dstHeight * 3)
        val scaleX = srcWidth.toDouble() / dstWidth
        val scaleY = srcHeight.toDouble() / dstHeight

        for (dy in 0 until dstHeight) {
            val sourceY = ((dy + 0.5) * scaleY - 0.5).coerceIn(0.0, (srcHeight - 1).toDouble())
            val y0 = sourceY.toInt()
            val y1 = minOf(y0 + 1, srcHeight - 1)
            val wy = sourceY - y0
            for (dx in 0 until dstWidth) {
                val sourceX = ((dx + 0.5) * scaleX - 0.5).coerceIn(0.0, (srcWidth - 1).toDouble())
                val x0 = sourceX.toInt()
                val x1 = minOf(x0 + 1, srcWidth - 1)
                val wx = sourceX - x0

                val base = (dy * dstWidth + dx) * 3
                for (channel in 0 until 3) {
                    val top = pixels[(y0 * srcWidth + x0) * 3 + channel] * (1.0 - wx) +
                        pixels[(y0 * srcWidth + x1) * 3 + channel] * wx
                    val bottom = pixels[(y1 * srcWidth + x0) * 3 + channel] * (1.0 - wx) +
                        pixels[(y1 * srcWidth + x1) * 3 + channel] * wx
                    val value = top * (1.0 - wy) + bottom * wy
                    // tf.cast truncates toward zero; values are non-negative here.
                    output[base + channel] = value.toInt().coerceIn(0, 255).toFloat()
                }
            }
        }
        return output
    }
}
