package com.kimhs.liverdx.ml

import kotlin.math.exp

/**
 * Cleanup applied to a phone photo before it reaches the model.
 *
 * Direct port of `android_app/tools/photo_pipeline.py`; `PhotoPreprocessTest` pins both to the same
 * fixture, so any change here has to be made there too. The model was trained on clean, effectively
 * grayscale SMC-LUD frames, so a phone photo needs five corrections, applied in this fixed order:
 *
 *  1. [STAGE_GLARE] — blown-out specular highlights. First, because illumination flattening rescales
 *     blown pixels back into range and would hide them from the detector.
 *  2. [STAGE_FLATTEN] — the large-scale brightness ramp from an unlit screen corner or a vignette.
 *  3. [STAGE_DESATURATE] — collapse the photo's colour cast to the grayscale training domain.
 *  4. [STAGE_DENOISE] — sensor grain and JPEG ringing, with a bilateral filter that keeps margins.
 *  5. [STAGE_CONTRAST] — exposure normalisation (1st-99th percentile).
 *
 * Every stage is switchable so the user can watch each one act on a given frame.
 */
object PhotoPreprocess {

    const val STAGE_GLARE = "remove_glare"
    const val STAGE_FLATTEN = "flatten_illumination"
    const val STAGE_DESATURATE = "desaturate"
    const val STAGE_DENOISE = "denoise"
    const val STAGE_CONTRAST = "stretch_contrast"

    val STAGE_ORDER = listOf(STAGE_GLARE, STAGE_FLATTEN, STAGE_DESATURATE, STAGE_DENOISE, STAGE_CONTRAST)

    private const val LUMA_RED = 0.299f
    private const val LUMA_GREEN = 0.587f
    private const val LUMA_BLUE = 0.114f
    private const val FLATTEN_GAIN_MIN = 0.5
    private const val FLATTEN_GAIN_MAX = 2.0
    private const val GLARE_ABSOLUTE = 238.0
    private const val GLARE_EXCESS = 45.0
    private const val GLARE_FLOOR = 215.0
    private const val GLARE_OPEN_RADIUS = 1
    private const val GLARE_HALO_RADIUS = 2
    private const val INPAINT_FILL_RADIUS_DIVISOR = 20
    private const val INPAINT_ITERATIONS = 48
    private const val NEIGHBOUR_COUNT = 4
    private const val BILATERAL_RADIUS = 2
    private const val BILATERAL_SPATIAL_SIGMA = 1.6
    private const val BILATERAL_SIGMA_RANGE = 18.0
    private const val CONTRAST_LOW_PERCENTILE = 0.01
    private const val CONTRAST_HIGH_PERCENTILE = 0.99
    private const val CONTRAST_MIN_SPAN = 8.0
    private const val HISTOGRAM_BINS = 256
    private const val FULL_VALUE = 255.0
    private const val EPSILON = 1e-6

    private val NEIGHBOURS = arrayOf(intArrayOf(-1, 0), intArrayOf(1, 0), intArrayOf(0, -1), intArrayOf(0, 1))
    private val BILATERAL_OFFSETS = buildOffsets()
    private val BILATERAL_SPATIAL_WEIGHTS = DoubleArray(BILATERAL_OFFSETS.size) { index ->
        val (dy, dx) = BILATERAL_OFFSETS[index]
        exp(-(dy * dy + dx * dx) / (2.0 * BILATERAL_SPATIAL_SIGMA * BILATERAL_SPATIAL_SIGMA))
    }

    /** Which stages to run. Defaults are the camera defaults. */
    data class Options(
        val removeGlare: Boolean = true,
        val flattenIllumination: Boolean = true,
        val desaturate: Boolean = true,
        val denoise: Boolean = true,
        val stretchContrast: Boolean = true,
    ) {
        val anyEnabled: Boolean
            get() = removeGlare || flattenIllumination || desaturate || denoise || stretchContrast

        val enabledStages: List<String>
            get() = STAGE_ORDER.filter(::isEnabled)

        /** Reads one stage flag by its metadata id. */
        fun isEnabled(stage: String): Boolean = when (stage) {
            STAGE_GLARE -> removeGlare
            STAGE_FLATTEN -> flattenIllumination
            STAGE_DESATURATE -> desaturate
            STAGE_DENOISE -> denoise
            STAGE_CONTRAST -> stretchContrast
            else -> false
        }

        /** Returns a copy with one stage flag written by its metadata id. */
        fun withStage(stage: String, enabled: Boolean): Options = when (stage) {
            STAGE_GLARE -> copy(removeGlare = enabled)
            STAGE_FLATTEN -> copy(flattenIllumination = enabled)
            STAGE_DESATURATE -> copy(desaturate = enabled)
            STAGE_DENOISE -> copy(denoise = enabled)
            STAGE_CONTRAST -> copy(stretchContrast = enabled)
            else -> this
        }
    }

    data class Result(val rgb: FloatArray, val glareFraction: Float, val appliedStages: List<String>)

    /** Runs the enabled stages in [STAGE_ORDER] over a row-major RGB frame in [0, 255]. */
    fun run(rgb: FloatArray, width: Int, height: Int, options: Options): Result {
        require(rgb.size == width * height * 3) {
            "expected ${width * height * 3} values for ${width}x$height, got ${rgb.size}"
        }
        var current = rgb.copyOf()
        var glareFraction = 0f

        for (stage in STAGE_ORDER) {
            when (stage) {
                STAGE_GLARE -> if (options.removeGlare) {
                    val mask = glareMask(current, width, height)
                    glareFraction = mask.count { it }.toFloat() / mask.size
                    current = inpaint(current, mask, width, height)
                }
                STAGE_FLATTEN -> if (options.flattenIllumination) {
                    current = flattenIllumination(current, width, height)
                }
                STAGE_DESATURATE -> if (options.desaturate) {
                    current = desaturate(current, width, height)
                }
                STAGE_DENOISE -> if (options.denoise) {
                    current = denoise(current, width, height)
                }
                STAGE_CONTRAST -> if (options.stretchContrast) {
                    current = stretchContrast(current, width, height)
                }
            }
        }

        for (index in current.indices) current[index] = current[index].coerceIn(0f, FULL_VALUE.toFloat())
        return Result(current, glareFraction, options.enabledStages)
    }

    /** Rec.601 luma, kept in float arithmetic to stay comparable with the numpy reference. */
    fun luma(rgb: FloatArray, width: Int, height: Int): FloatArray {
        val out = FloatArray(width * height)
        for (index in out.indices) {
            val base = index * 3
            out[index] = rgb[base] * LUMA_RED + rgb[base + 1] * LUMA_GREEN + rgb[base + 2] * LUMA_BLUE
        }
        return out
    }

    /** Blown highlights, opened to spare thin anatomy and then grown back with a one-pixel halo. */
    fun glareMask(rgb: FloatArray, width: Int, height: Int): BooleanArray {
        val radius = maxOf(3, roundHalfUp(minOf(width, height) / 20.0))
        val luminance = luma(rgb, width, height)
        val background = boxBlur(luminance, width, height, radius)

        val raw = BooleanArray(luminance.size)
        for (index in luminance.indices) {
            val value = luminance[index].toDouble()
            val isClipped = value >= GLARE_ABSOLUTE
            val isLocalExcess = (value - background[index]) >= GLARE_EXCESS && value >= GLARE_FLOOR
            raw[index] = isClipped || isLocalExcess
        }
        return dilate(erode(raw, width, height, GLARE_OPEN_RADIUS), width, height, GLARE_HALO_RADIUS)
    }

    /**
     * Fills every masked pixel from the local mean of the unmasked ones, then diffuses the seams.
     *
     * Diffusion alone would need O(radius^2) sweeps to reach the middle of a blob, so the interior is
     * seeded first and only then smoothed.
     */
    fun inpaint(rgb: FloatArray, mask: BooleanArray, width: Int, height: Int): FloatArray {
        val masked = ArrayList<Int>()
        for (index in mask.indices) if (mask[index]) masked.add(index)
        if (masked.isEmpty()) return rgb.copyOf()

        val radius = maxOf(2, roundHalfUp(minOf(width, height) / INPAINT_FILL_RADIUS_DIVISOR.toDouble()))
        val valid = DoubleArray(mask.size) { if (mask[it]) 0.0 else 1.0 }
        val neighbourhood = boxBlur(valid, width, height, radius)

        var current = rgb.copyOf()
        for (channel in 0 until 3) {
            val contribution = DoubleArray(valid.size)
            for (index in valid.indices) {
                if (valid[index] > 0.0) contribution[index] = rgb[index * 3 + channel].toDouble()
            }
            val blurred = boxBlur(contribution, width, height, radius)
            for (entry in masked) {
                val weight = maxOf(neighbourhood[entry], EPSILON)
                current[entry * 3 + channel] = (blurred[entry] / weight).coerceIn(0.0, FULL_VALUE).toFloat()
            }
        }

        var next = current.copyOf()
        val rows = IntArray(masked.size) { masked[it] / width }
        val cols = IntArray(masked.size) { masked[it] % width }
        repeat(INPAINT_ITERATIONS) {
            System.arraycopy(current, 0, next, 0, current.size)
            for (entry in masked.indices) {
                val y = rows[entry]
                val x = cols[entry]
                val base = masked[entry] * 3
                for (channel in 0 until 3) {
                    var total = 0.0
                    for (neighbour in NEIGHBOURS) {
                        val ny = (y + neighbour[0]).coerceIn(0, height - 1)
                        val nx = (x + neighbour[1]).coerceIn(0, width - 1)
                        total += current[(ny * width + nx) * 3 + channel]
                    }
                    next[base + channel] = (total / NEIGHBOUR_COUNT).toFloat()
                }
            }
            val swap = current
            current = next
            next = swap
        }
        return current
    }

    private fun flattenIllumination(rgb: FloatArray, width: Int, height: Int): FloatArray {
        val radius = maxOf(4, roundHalfUp(minOf(width, height) / 12.0))
        val luminance = luma(rgb, width, height)
        val background = boxBlur(luminance, width, height, radius)

        var sum = 0.0
        for (value in luminance) sum += value
        val mean = sum / luminance.size

        val out = FloatArray(rgb.size)
        for (index in luminance.indices) {
            val gain = (mean / maxOf(background[index], 1e-3)).coerceIn(FLATTEN_GAIN_MIN, FLATTEN_GAIN_MAX)
            val base = index * 3
            for (channel in 0 until 3) {
                out[base + channel] = (rgb[base + channel] * gain).coerceIn(0.0, FULL_VALUE).toFloat()
            }
        }
        return out
    }

    private fun desaturate(rgb: FloatArray, width: Int, height: Int): FloatArray {
        val luminance = luma(rgb, width, height)
        val out = FloatArray(rgb.size)
        for (index in luminance.indices) {
            val base = index * 3
            out[base] = luminance[index]
            out[base + 1] = luminance[index]
            out[base + 2] = luminance[index]
        }
        return out
    }

    /** 5x5 bilateral filter: edge-preserving, so lesion margins survive while grain does not. */
    private fun denoise(rgb: FloatArray, width: Int, height: Int): FloatArray {
        val rangeDenominator = 2.0 * BILATERAL_SIGMA_RANGE * BILATERAL_SIGMA_RANGE
        val out = FloatArray(rgb.size)

        for (y in 0 until height) {
            for (x in 0 until width) {
                val centreBase = (y * width + x) * 3
                val total = DoubleArray(3)
                var weightSum = 0.0

                for (offset in BILATERAL_OFFSETS.indices) {
                    val (dy, dx) = BILATERAL_OFFSETS[offset]
                    val ny = y + dy
                    val nx = x + dx
                    if (ny < 0 || ny >= height || nx < 0 || nx >= width) continue

                    val neighbourBase = (ny * width + nx) * 3
                    var squaredDifference = 0.0
                    for (channel in 0 until 3) {
                        val difference = rgb[neighbourBase + channel].toDouble() - rgb[centreBase + channel]
                        squaredDifference += difference * difference
                    }
                    val weight = BILATERAL_SPATIAL_WEIGHTS[offset] * exp(-squaredDifference / rangeDenominator)
                    for (channel in 0 until 3) total[channel] += rgb[neighbourBase + channel] * weight
                    weightSum += weight
                }

                for (channel in 0 until 3) {
                    val value = if (weightSum > 0.0) total[channel] / weightSum else rgb[centreBase + channel].toDouble()
                    out[centreBase + channel] = value.coerceIn(0.0, FULL_VALUE).toFloat()
                }
            }
        }
        return out
    }

    /** 1-99 percentile stretch on 256 unit bins built by truncation, so both ports agree exactly. */
    private fun stretchContrast(rgb: FloatArray, width: Int, height: Int): FloatArray {
        val luminance = luma(rgb, width, height)
        val histogram = IntArray(HISTOGRAM_BINS)
        for (value in luminance) histogram[value.toInt().coerceIn(0, HISTOGRAM_BINS - 1)]++

        val total = histogram.sum()
        if (total <= 0) return rgb.copyOf()

        val low = firstBinAt(histogram, CONTRAST_LOW_PERCENTILE * total, default = 0)
        val high = firstBinAt(histogram, CONTRAST_HIGH_PERCENTILE * total, default = HISTOGRAM_BINS - 1)
        val span = (high - low).toDouble()
        if (span < CONTRAST_MIN_SPAN) return rgb.copyOf()

        val scale = FULL_VALUE / span
        val out = FloatArray(rgb.size)
        for (index in rgb.indices) {
            out[index] = ((rgb[index] - low) * scale).coerceIn(0.0, FULL_VALUE).toFloat()
        }
        return out
    }

    fun boxBlur(field: FloatArray, width: Int, height: Int, radius: Int): DoubleArray {
        val asDoubles = DoubleArray(field.size) { field[it].toDouble() }
        return boxBlur(asDoubles, width, height, radius)
    }

    /** Mean over a (2r+1)^2 window through a float64 integral image. */
    fun boxBlur(field: DoubleArray, width: Int, height: Int, radius: Int): DoubleArray {
        val stride = width + 1
        val integral = DoubleArray(stride * (height + 1))
        for (y in 0 until height) {
            var rowSum = 0.0
            val rowOffset = y * width
            val integralRow = (y + 1) * stride
            val previousRow = y * stride
            for (x in 0 until width) {
                rowSum += field[rowOffset + x]
                integral[integralRow + x + 1] = integral[previousRow + x + 1] + rowSum
            }
        }

        val out = DoubleArray(field.size)
        for (y in 0 until height) {
            val top = (y - radius).coerceAtLeast(0)
            val bottom = (y + radius + 1).coerceAtMost(height)
            val innerRows = (bottom - top).toDouble()
            val topRow = top * stride
            val bottomRow = bottom * stride
            for (x in 0 until width) {
                val left = (x - radius).coerceAtLeast(0)
                val right = (x + radius + 1).coerceAtMost(width)
                val total = integral[bottomRow + right] - integral[topRow + right] -
                    integral[bottomRow + left] + integral[topRow + left]
                out[y * width + x] = total / (innerRows * (right - left))
            }
        }
        return out
    }

    private fun erode(mask: BooleanArray, width: Int, height: Int, radius: Int): BooleanArray {
        val out = BooleanArray(mask.size) { true }
        for (y in 0 until height) {
            for (x in 0 until width) {
                var keep = true
                for (dy in -radius..radius) {
                    if (!keep) break
                    val ny = (y + dy).coerceIn(0, height - 1)
                    for (dx in -radius..radius) {
                        val nx = (x + dx).coerceIn(0, width - 1)
                        if (!mask[ny * width + nx]) {
                            keep = false
                            break
                        }
                    }
                }
                out[y * width + x] = keep
            }
        }
        return out
    }

    private fun dilate(mask: BooleanArray, width: Int, height: Int, radius: Int): BooleanArray {
        val out = BooleanArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var grow = false
                for (dy in -radius..radius) {
                    if (grow) break
                    val ny = (y + dy).coerceIn(0, height - 1)
                    for (dx in -radius..radius) {
                        val nx = (x + dx).coerceIn(0, width - 1)
                        if (mask[ny * width + nx]) {
                            grow = true
                            break
                        }
                    }
                }
                out[y * width + x] = grow
            }
        }
        return out
    }

    private fun firstBinAt(histogram: IntArray, target: Double, default: Int): Int {
        var cumulative = 0
        for (index in histogram.indices) {
            cumulative += histogram[index]
            if (cumulative >= target) return index
        }
        return default
    }

    private fun roundHalfUp(value: Double): Int = (value + 0.5).toInt()

    private fun buildOffsets(): Array<Pair<Int, Int>> = buildList {
        for (dy in -BILATERAL_RADIUS..BILATERAL_RADIUS) {
            for (dx in -BILATERAL_RADIUS..BILATERAL_RADIUS) add(dy to dx)
        }
    }.toTypedArray()
}
