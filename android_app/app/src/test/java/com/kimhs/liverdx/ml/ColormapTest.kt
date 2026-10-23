package com.kimhs.liverdx.ml

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ColormapTest {

    @Test
    fun `anchors are returned exactly at their positions`() {
        Colormap.INFERNO.indices.forEach { index ->
            val t = index.toFloat() / (Colormap.INFERNO.size - 1)
            assertEquals(Colormap.INFERNO[index], Colormap.inferno(t))
        }
        Colormap.TURBO.indices.forEach { index ->
            val t = index.toFloat() / (Colormap.TURBO.size - 1)
            assertEquals(Colormap.TURBO[index], Colormap.turbo(t))
        }
    }

    @Test
    fun `sampling clamps outside the unit interval`() {
        assertEquals(Colormap.INFERNO.first(), Colormap.inferno(-3f))
        assertEquals(Colormap.INFERNO.last(), Colormap.inferno(4f))
    }

    @Test
    fun `every sample is opaque and interpolates in hue`() {
        val low = Colormap.turbo(0f)
        val mid = Colormap.turbo(0.5f)
        val high = Colormap.turbo(1f)
        for (sample in listOf(low, mid, high)) {
            assertEquals(0xFF, (sample shr 24) and 0xFF)
        }
        assertNotEquals(low, mid)
        assertNotEquals(mid, high)
        assertTrue("ramp start should be blue-dominant", value(low, 0) > value(low, 16))
        assertTrue("ramp end should be red-dominant", value(high, 16) > value(high, 0))
    }

    @Test
    fun `blending is linear per channel`() {
        assertEquals(0xFF000000.toInt(), Colormap.lerpArgb(0xFF000000.toInt(), 0xFFFFFFFF.toInt(), 0f))
        assertEquals(0xFFFFFFFF.toInt(), Colormap.lerpArgb(0xFF000000.toInt(), 0xFFFFFFFF.toInt(), 1f))
        assertEquals(0xFF808080.toInt(), Colormap.lerpArgb(0xFF000000.toInt(), 0xFFFFFFFF.toInt(), 0.5f))
    }

    private fun value(argb: Int, shift: Int): Int = (argb shr shift) and 0xFF
}
