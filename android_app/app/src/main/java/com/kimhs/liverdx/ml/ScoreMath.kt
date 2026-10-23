package com.kimhs.liverdx.ml

/**
 * Pure-Kotlin numeric core: image resampling, L2/cosine geometry and map normalisation.
 *
 * Deliberately free of Android types so it can run under `./gradlew :app:testDebugUnitTest` and be
 * checked against `app/src/test/resources/python_reference.json`, which is produced by
 * `tools/export_tflite.py` using the same formulas (numpy + cv2.INTER_AREA).
 */
object ScoreMath {

    /** L2-normalises [values] into a new array. Zero vectors are returned unchanged. */
    fun l2Normalize(values: FloatArray): FloatArray {
        var sum = 0.0
        for (value in values) sum += value.toDouble() * value.toDouble()
        val norm = kotlin.math.sqrt(sum)
        if (norm < 1e-12) return values.copyOf()
        val scale = (1.0 / norm).toFloat()
        return FloatArray(values.size) { values[it] * scale }
    }

    /**
     * Cosine similarity between `values[offset until offset+other.size]` and [other].
     * Both are normalised here, so callers may pass raw embeddings.
     */
    fun cosine(values: FloatArray, other: FloatArray, offset: Int = 0): Float {
        var dot = 0.0
        var normA = 0.0
        var normB = 0.0
        for (i in other.indices) {
            val a = values[offset + i].toDouble()
            val b = other[i].toDouble()
            dot += a * b
            normA += a * a
            normB += b * b
        }
        val denom = kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB)
        return if (denom < 1e-12) 0f else (dot / denom).toFloat()
    }

    /** Rescales [values] into [0, 1] by min-max. A constant input maps to all zeros. */
    fun minMaxNormalize(values: FloatArray): FloatArray {
        var min = Float.MAX_VALUE
        var max = -Float.MAX_VALUE
        for (value in values) {
            if (value < min) min = value
            if (value > max) max = value
        }
        val span = max - min
        if (span < 1e-12f) return FloatArray(values.size)
        return FloatArray(values.size) { (values[it] - min) / span }
    }

    /**
     * Per-patch mean cosine similarity to a prototype bank, min-max normalised to [0, 1].
     *
     * @param patchTokens row-major `[grid * grid, prototypes[0].size]`
     * @param prototypes  prototype bank, e.g. the k-means HCC bank `[4, 128]`
     * @return row-major `grid * grid` map, ready for [Colormap]
     */
    fun prototypeMap(patchTokens: FloatArray, prototypes: Array<FloatArray>, grid: Int): FloatArray {
        require(prototypes.isNotEmpty()) { "prototype bank is empty" }
        val dim = prototypes.first().size
        val patches = patchTokens.size / dim
        require(patches == grid * grid) { "patch tokens $patches do not fill a $grid x $grid grid" }
        val perPatch = FloatArray(patches)
        for (patch in 0 until patches) {
            val offset = patch * dim
            var sum = 0.0
            for (prototype in prototypes) sum += cosine(patchTokens, prototype, offset)
            perPatch[patch] = (sum / prototypes.size).toFloat()
        }
        return minMaxNormalize(perPatch)
    }

    /**
     * Area-average downscale over ARGB_8888 pixels, as returned by `Bitmap.getPixels`.
     *
     * The arithmetic lives in [Resampler.areaAverage]; this overload only unpacks the pixels.
     */
    fun resizeAreaAvg(pixels: IntArray, srcWidth: Int, srcHeight: Int, dstWidth: Int, dstHeight: Int): FloatArray {
        require(pixels.size >= srcWidth * srcHeight) {
            "pixel buffer holds ${pixels.size} values, need ${srcWidth * srcHeight}"
        }
        val pixelCount = srcWidth * srcHeight
        val rgb = FloatArray(pixelCount * 3)
        for (index in 0 until pixelCount) {
            val argb = pixels[index]
            val base = index * 3
            rgb[base] = ((argb shr 16) and 0xFF).toFloat()
            rgb[base + 1] = ((argb shr 8) and 0xFF).toFloat()
            rgb[base + 2] = (argb and 0xFF).toFloat()
        }
        return Resampler.areaAverage(rgb, srcWidth, srcHeight, dstWidth, dstHeight)
    }
}
