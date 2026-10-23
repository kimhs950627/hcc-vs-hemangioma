package com.kimhs.liverdx.ml

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/**
 * Verifies the pure-Kotlin numeric core against `python_reference.json`, which
 * `android_app/tools/export_tflite.py` generates from the same formulas used in the paper code
 * (numpy / cv2) and from the real TFLite flatbuffer output on `example_images_hcc_case`.
 */
class ScoreMathTest {

    private val reference: JSONObject by lazy {
        val stream = javaClass.getResourceAsStream("/python_reference.json")
            ?: error("python_reference.json missing — run: python android_app/tools/export_tflite.py")
        JSONObject(stream.bufferedReader().use { it.readText() })
    }

    @Test
    fun `flatbuffer export stayed within one ten-thousandth of Keras`() {
        val report = reference.getJSONObject("max_abs_diff_vs_keras")
        val worst = report.getDouble("cosine")
        assertTrue("worst |TFLite - Keras| cosine drift was $worst, expected < 1e-3", worst < 1e-3)
        assertEquals("fixture must describe the fp32 model", "fp32", reference.getString("precision"))
    }

    @Test
    fun `area average downscale matches the reference implementation`() {
        val fixture = reference.getJSONObject("resize_case")
        val srcWidth = fixture.getInt("src_width")
        val srcHeight = fixture.getInt("src_height")
        val dstWidth = fixture.getInt("dst_width")
        val dstHeight = fixture.getInt("dst_height")
        val tolerance = fixture.getDouble("tolerance")

        val rgb = fixture.getJSONArray("src_pixels_rgb")
        val pixelCount = srcWidth * srcHeight
        assertEquals("fixture RGB triples", pixelCount * 3, rgb.length())
        // The fixture stores row-major RGB triplets; Bitmap.getPixels wants one ARGB int per pixel.
        val pixels = IntArray(pixelCount) { index ->
            val r = rgb.getInt(index * 3)
            val g = rgb.getInt(index * 3 + 1)
            val b = rgb.getInt(index * 3 + 2)
            (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }

        val actual = ScoreMath.resizeAreaAvg(pixels, srcWidth, srcHeight, dstWidth, dstHeight)
        val expected = fixture.getJSONArray("expected_rgb").toFloatArray()
        assertEquals(dstWidth * dstHeight * 3, actual.size)

        var worst = 0.0
        for (index in actual.indices) worst = maxOf(worst, abs(actual[index] - expected[index]).toDouble())
        assertTrue("max |kotlin - reference| was $worst, expected <= $tolerance", worst <= tolerance)

        // The same numbers rounded to uint8 must reproduce cv2.INTER_AREA exactly, up to cvRound.
        val rounded = fixture.getJSONArray("opencv_rounded_rgb").toFloatArray()
        val roundingTolerance = fixture.getDouble("opencv_rounding_tolerance")
        var worstRounded = 0.0
        for (index in actual.indices) {
            worstRounded = maxOf(worstRounded, abs(actual[index] - rounded[index]).toDouble())
        }
        assertTrue(
            "max |kotlin - cv2.INTER_AREA| was $worstRounded, expected <= $roundingTolerance",
            worstRounded <= roundingTolerance,
        )
    }

    @Test
    fun `cosine scores reproduce the python predictor on every example case`() {
        val prototypes = reference.getJSONObject("prototypes")
        val hccMean = prototypes.getJSONArray("hcc_mean").toFloatMatrix().first()
        val hemMean = prototypes.getJSONArray("hem_mean").toFloatMatrix().first()
        val cases = reference.getJSONArray("cases")
        assertTrue("fixture carries no example cases", cases.length() > 0)

        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val name = case.getString("name")
            val embedding = case.getJSONArray("embedding").toFloatArray()

            val hccCosine = ScoreMath.cosine(embedding, hccMean)
            val hemCosine = ScoreMath.cosine(embedding, hemMean)
            val delta = hccCosine - hemCosine

            assertEquals("$name hcc cosine", case.getDouble("hcc_cosine_score"), hccCosine.toDouble(), COSINE_TOLERANCE)
            assertEquals("$name hemangioma cosine", case.getDouble("hem_cosine_score"), hemCosine.toDouble(), COSINE_TOLERANCE)
            assertEquals("$name delta", case.getDouble("delta_score"), delta.toDouble(), COSINE_TOLERANCE)
        }
    }

    @Test
    fun `prototype map matches the python formula on a synthetic patch grid`() {
        val fixture = reference.getJSONObject("prototype_map_case")
        val grid = fixture.getInt("grid")
        val tokens = fixture.getJSONArray("patch_tokens").toFloatArray()
        val prototypes = fixture.getJSONArray("prototypes").toFloatMatrix()
        val expected = fixture.getJSONArray("expected").toFloatArray()
        val tolerance = fixture.getDouble("tolerance")

        val actual = ScoreMath.prototypeMap(tokens, prototypes, grid)
        assertEquals(grid * grid, actual.size)
        for (index in actual.indices) {
            assertEquals("patch $index", expected[index].toDouble(), actual[index].toDouble(), tolerance)
        }
    }

    @Test
    fun `prototype map reproduces the python prototype map on a real frame`() {
        val cases = reference.getJSONArray("cases")
        val prototypes = reference.getJSONObject("prototypes").getJSONArray("hcc_kmeans").toFloatMatrix()
        var checked = 0

        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            if (!case.has("patch_tokens")) continue
            checked++
            val grid = 12
            val tokens = case.getJSONArray("patch_tokens").toFloatArray()
            val expected = case.getJSONArray("hcc_prototype_map").toFloatArray()
            val actual = ScoreMath.prototypeMap(tokens, prototypes, grid)
            for (position in actual.indices) {
                assertEquals(
                    "${case.getString("name")} patch $position",
                    expected[position].toDouble(),
                    actual[position].toDouble(),
                    PROTOTYPE_MAP_TOLERANCE,
                )
            }
        }
        assertTrue("no case carried patch tokens", checked > 0)
    }

    @Test
    fun `attention normalisation reproduces the exported display map`() {
        val cases = reference.getJSONArray("cases")
        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val raw = case.getJSONArray("att_map_raw").toFloatArray()
            val exported = case.getJSONArray("att_map").toFloatArray()
            val actual = ScoreMath.minMaxNormalize(raw)
            for (position in actual.indices) {
                assertEquals(
                    "${case.getString("name")} cell $position",
                    exported[position].toDouble(),
                    actual[position].toDouble(),
                    ATTENTION_TOLERANCE,
                )
            }
        }
    }

    private fun JSONArray.toFloatArray(): FloatArray =
        FloatArray(length()) { getDouble(it).toFloat() }

    private fun JSONArray.toFloatMatrix(): Array<FloatArray> =
        Array(length()) { row -> getJSONArray(row).toFloatArray() }

    private companion object {
        const val COSINE_TOLERANCE = 1e-4
        const val PROTOTYPE_MAP_TOLERANCE = 1e-4
        const val ATTENTION_TOLERANCE = 1e-5
    }
}
