package com.kimhs.liverdx.ml

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Both resamplers are pinned to their Python counterparts through `python_reference.json`:
 *  - `bilinear_case` is `tf.image.resize(bilinear)` + `tf.cast(uint8)`, i.e. the dataloader path the
 *    reported metrics came from,
 *  - `resize_case` is `cv2.INTER_AREA`, the explainable_predictor path.
 */
class ResamplerTest {

    private val reference: JSONObject by lazy {
        val stream = javaClass.getResourceAsStream("/python_reference.json")
            ?: error("python_reference.json missing — run: python android_app/tools/export_tflite.py")
        JSONObject(stream.bufferedReader().use { it.readText() })
    }

    @Test
    fun `dataloader bilinear plus uint8 matches tf image resize exactly`() {
        assertResamplerMatches("bilinear_case", ResizeMode.BILINEAR_UINT8)
    }

    @Test
    fun `area average matches the cv2 reference`() {
        assertResamplerMatches("resize_case", ResizeMode.AREA)
    }

    @Test
    fun `the two modes actually disagree, so the switch is not cosmetic`() {
        val resolved = resolve("bilinear_case")
        val bilinear = Resampler.resample(
            resolved.rgb, resolved.srcWidth, resolved.srcHeight,
            resolved.dstWidth, resolved.dstHeight, ResizeMode.BILINEAR_UINT8,
        )
        val area = Resampler.resample(
            resolved.rgb, resolved.srcWidth, resolved.srcHeight,
            resolved.dstWidth, resolved.dstHeight, ResizeMode.AREA,
        )
        var worst = 0.0
        for (index in bilinear.indices) worst = maxOf(worst, abs(bilinear[index] - area[index]).toDouble())
        assertTrue("bilinear and area produced identical output; fixture cannot detect a mode swap", worst > 1.0)
    }

    private fun assertResamplerMatches(fixtureName: String, mode: ResizeMode) {
        val fixture = reference.getJSONObject(fixtureName)
        val resolved = resolve(fixtureName)
        val expected = fixture.getJSONArray("expected_rgb").toFloatArray()
        val tolerance = fixture.getDouble("tolerance")

        val actual = Resampler.resample(
            resolved.rgb, resolved.srcWidth, resolved.srcHeight,
            resolved.dstWidth, resolved.dstHeight, mode,
        )
        assertEquals(resolved.dstWidth * resolved.dstHeight * 3, actual.size)

        var worst = 0.0
        for (index in actual.indices) worst = maxOf(worst, abs(actual[index] - expected[index]).toDouble())
        assertTrue("$fixtureName (${mode.id}) max diff was $worst, expected <= $tolerance", worst <= tolerance)
    }

    private class Resolved(
        val rgb: FloatArray,
        val srcWidth: Int,
        val srcHeight: Int,
        val dstWidth: Int,
        val dstHeight: Int,
    )

    private fun resolve(fixtureName: String): Resolved {
        val fixture = reference.getJSONObject(fixtureName)
        val srcWidth = fixture.getInt("src_width")
        val srcHeight = fixture.getInt("src_height")
        val triples = fixture.getJSONArray("src_pixels_rgb").toFloatArray()
        assertEquals("fixture RGB triples", srcWidth * srcHeight * 3, triples.size)
        return Resolved(
            rgb = triples,
            srcWidth = srcWidth,
            srcHeight = srcHeight,
            dstWidth = fixture.getInt("dst_width"),
            dstHeight = fixture.getInt("dst_height"),
        )
    }

    private fun JSONArray.toFloatArray(): FloatArray = FloatArray(length()) { getDouble(it).toFloat() }
}
