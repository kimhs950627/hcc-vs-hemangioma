package com.kimhs.liverdx.ml

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Pins the Kotlin cleanup to the numpy reference in `android_app/tools/photo_pipeline.py`.
 *
 * The synthetic frame in the fixture carries all four defects the pipeline exists for (blown glare,
 * a backlight ramp, a colour cast, grain), and the fixture exercises three option sets so the toggles
 * are covered too, including a fully disabled pass that must be the identity.
 */
class PhotoPreprocessTest {

    private val reference: JSONObject by lazy {
        val stream = javaClass.getResourceAsStream("/python_reference.json")
            ?: error("python_reference.json missing — run: python android_app/tools/export_tflite.py")
        JSONObject(stream.bufferedReader().use { it.readText() })
    }

    private val fixture: JSONObject get() = reference.getJSONObject("preprocess_case")

    @Test
    fun `every option set reproduces the numpy reference`() {
        val width = fixture.getInt("width")
        val height = fixture.getInt("height")
        val source = fixture.getJSONArray("src_rgb").toFloatArray()
        val cases = fixture.getJSONObject("cases")
        val tolerance = fixture.getDouble("tolerance")

        assertEquals(width * height * 3, source.size)
        for (name in cases.keys()) {
            val case = cases.getJSONObject(name)
            val options = case.getJSONObject("options")
            val result = PhotoPreprocess.run(
                source,
                width,
                height,
                PhotoPreprocess.Options(
                    removeGlare = options.getBoolean("remove_glare"),
                    flattenIllumination = options.getBoolean("flatten_illumination"),
                    desaturate = options.getBoolean("desaturate"),
                    denoise = options.getBoolean("denoise"),
                    stretchContrast = options.getBoolean("stretch_contrast"),
                ),
            )

            val expected = case.getJSONArray("expected_rgb").toFloatArray()
            var worst = 0.0
            for (index in result.rgb.indices) {
                worst = maxOf(worst, abs(result.rgb[index] - expected[index]).toDouble())
            }
            assertTrue("[$name] max |kotlin - numpy| was $worst, expected <= $tolerance", worst <= tolerance)

            val expectedGlare = case.getDouble("expected_glare_fraction")
            assertEquals("[$name] glare fraction", expectedGlare, result.glareFraction.toDouble(), GLARE_TOLERANCE)
            assertEquals("[$name] applied stages", options.enabledCount(), result.appliedStages.size)
        }
    }

    @Test
    fun `disabling every stage returns the frame untouched`() {
        val width = fixture.getInt("width")
        val height = fixture.getInt("height")
        val source = fixture.getJSONArray("src_rgb").toFloatArray()
        val result = PhotoPreprocess.run(
            source,
            width,
            height,
            PhotoPreprocess.Options(
                removeGlare = false,
                flattenIllumination = false,
                desaturate = false,
                denoise = false,
                stretchContrast = false,
            ),
        )
        assertTrue("no stage should report as applied", result.appliedStages.isEmpty())
        assertEquals(0f, result.glareFraction)
        for (index in source.indices) {
            assertEquals("value $index", source[index].toDouble(), result.rgb[index].toDouble(), 1e-6)
        }
    }

    @Test
    fun `glare detection finds the blown highlight and nothing else`() {
        val width = fixture.getInt("width")
        val height = fixture.getInt("height")
        val source = fixture.getArray("src_rgb")
        val mask = PhotoPreprocess.glareMask(source, width, height)
        val fraction = mask.count { it }.toDouble() / mask.size

        // The synthetic highlight covers a small disc plus its one-pixel halo, not the whole frame.
        assertTrue("glare fraction was $fraction, expected something small but non-zero", fraction in 0.01..0.4)

        // The darkest corner must survive: the mask has to be about the highlight, not the exposure.
        val darkest = source.indices.minByOrNull { source[it] }!! / 3
        assertFalse("mask covered the darkest pixel", mask[darkest])
    }

    @Test
    fun `stage order is pinned`() {
        // Golden value: glare must come first (illumination flattening would otherwise rescale blown
        // pixels back into range and hide them) and any reorder has to regenerate the numpy fixture.
        assertEquals(
            listOf(
                PhotoPreprocess.STAGE_GLARE,
                PhotoPreprocess.STAGE_FLATTEN,
                PhotoPreprocess.STAGE_DESATURATE,
                PhotoPreprocess.STAGE_DENOISE,
                PhotoPreprocess.STAGE_CONTRAST,
            ),
            PhotoPreprocess.STAGE_ORDER,
        )
    }

    private fun JSONObject.enabledCount(): Int =
        listOf("remove_glare", "flatten_illumination", "desaturate", "denoise", "stretch_contrast")
            .count { getBoolean(it) }

    private fun JSONArray.toFloatArray(): FloatArray = FloatArray(length()) { getDouble(it).toFloat() }

    private fun JSONObject.getArray(name: String): FloatArray = getJSONArray(name).toFloatArray()

    private companion object {
        const val GLARE_TOLERANCE = 0.002
    }
}
