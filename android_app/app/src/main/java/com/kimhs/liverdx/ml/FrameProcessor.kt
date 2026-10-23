package com.kimhs.liverdx.ml

import android.graphics.Bitmap

/**
 * Turns a decoded bitmap into the tensor the model wants, and keeps the cleaned frame for display.
 *
 * Split out of [Classifier] so the model class stays a pure forward pass and this class owns the
 * whole input contract: pixel extraction, the optional photo cleanup, then the resample.
 */
class FrameProcessor(private val inputWidth: Int, private val inputHeight: Int) {

    /** @param modelInput row-major RGB in [0, 255], `inputWidth x inputHeight`, ready for TFLite */
    data class Outcome(
        val modelInput: FloatArray,
        val cleaned: FloatArray,
        val sourceWidth: Int,
        val sourceHeight: Int,
        val glareFraction: Float,
        val appliedStages: List<String>,
    ) {
        /** True when at least one cleanup stage actually changed the frame. */
        val cleanedApplied: Boolean get() = appliedStages.isNotEmpty()
    }

    fun process(
        source: Bitmap,
        options: PhotoPreprocess.Options,
        resizeMode: ResizeMode,
    ): Outcome {
        val rgb = extractRgb(source)
        val result = if (options.anyEnabled) {
            PhotoPreprocess.run(rgb, source.width, source.height, options)
        } else {
            PhotoPreprocess.Result(rgb, glareFraction = 0f, appliedStages = emptyList())
        }
        val modelInput = Resampler.resample(
            result.rgb,
            source.width,
            source.height,
            inputWidth,
            inputHeight,
            resizeMode,
        )
        return Outcome(
            modelInput = modelInput,
            cleaned = result.rgb,
            sourceWidth = source.width,
            sourceHeight = source.height,
            glareFraction = result.glareFraction,
            appliedStages = result.appliedStages,
        )
    }

    /** Row-major RGB floats in [0, 255] from an ARGB_8888 bitmap. */
    fun extractRgb(bitmap: Bitmap): FloatArray {
        val pixelCount = bitmap.width * bitmap.height
        val pixels = IntArray(pixelCount)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        val rgb = FloatArray(pixelCount * 3)
        for (index in 0 until pixelCount) {
            val argb = pixels[index]
            val base = index * 3
            rgb[base] = ((argb shr 16) and 0xFF).toFloat()
            rgb[base + 1] = ((argb shr 8) and 0xFF).toFloat()
            rgb[base + 2] = (argb and 0xFF).toFloat()
        }
        return rgb
    }

    companion object {
        /** Back to a bitmap so the before/after viewer can show the cleaned frame. */
        fun toBitmap(rgb: FloatArray, width: Int, height: Int): Bitmap {
            require(rgb.size == width * height * 3) {
                "expected ${width * height * 3} values for ${width}x$height, got ${rgb.size}"
            }
            val pixels = IntArray(width * height)
            for (index in pixels.indices) {
                val base = index * 3
                val red = rgb[base].toInt().coerceIn(0, 255)
                val green = rgb[base + 1].toInt().coerceIn(0, 255)
                val blue = rgb[base + 2].toInt().coerceIn(0, 255)
                pixels[index] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
            }
            return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        }
    }
}
