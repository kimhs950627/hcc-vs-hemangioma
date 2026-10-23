package com.kimhs.liverdx.ml

import android.graphics.Bitmap
import kotlin.math.roundToInt

/** Turns normalised 12x12 maps into display bitmaps and blends them over the ultrasound frame. */
object HeatmapRenderer {

    /**
     * Upsamples a normalised [grid] x [grid] map to a [width] x [height] bitmap through a colormap.
     *
     * The map covers the same frame the model saw, so it must be stretched to the preview's own
     * aspect ratio (the model squeezes the frame to a square, it does not letterbox it).
     */
    fun heatmapBitmap(values: FloatArray, grid: Int, width: Int, height: Int, colormap: (Float) -> Int): Bitmap {
        require(values.size == grid * grid) { "map holds ${values.size} values, expected ${grid * grid}" }
        val small = Bitmap.createBitmap(grid, grid, Bitmap.Config.ARGB_8888)
        small.setPixels(IntArray(grid * grid) { colormap(values[it]) }, 0, grid, 0, 0, grid, grid)

        val scaled = Bitmap.createScaledBitmap(small, width, height, true)
        if (scaled !== small) small.recycle()
        return scaled
    }

    /**
     * Source-over blend of [overlay] onto [base] at [alpha] in [0, 1].
     * [base] and [overlay] must share their dimensions.
     */
    fun blend(base: Bitmap, overlay: Bitmap, alpha: Float): Bitmap {
        require(base.width == overlay.width && base.height == overlay.height) {
            "cannot blend ${base.width}x${base.height} with ${overlay.width}x${overlay.height}"
        }
        val count = base.width * base.height
        val basePixels = IntArray(count)
        val overlayPixels = IntArray(count)
        base.getPixels(basePixels, 0, base.width, 0, 0, base.width, base.height)
        overlay.getPixels(overlayPixels, 0, base.width, 0, 0, base.width, overlay.height)

        val clamped = alpha.coerceIn(0f, 1f)
        val mixed = IntArray(count) { index -> composite(basePixels[index], overlayPixels[index], clamped) }
        return Bitmap.createBitmap(mixed, base.width, base.height, Bitmap.Config.ARGB_8888)
    }

    /** Interpolates every channel; the overlay is a full-opacity colormap so no alpha math is needed. */
    private fun composite(base: Int, overlay: Int, alpha: Float): Int {
        val inverse = 1f - alpha
        val red = (((overlay shr 16) and 0xFF) * alpha + ((base shr 16) and 0xFF) * inverse).roundToInt()
        val green = (((overlay shr 8) and 0xFF) * alpha + ((base shr 8) and 0xFF) * inverse).roundToInt()
        val blue = ((overlay and 0xFF) * alpha + (base and 0xFF) * inverse).roundToInt()
        return (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
    }
}
