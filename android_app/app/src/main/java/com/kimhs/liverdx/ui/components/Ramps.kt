package com.kimhs.liverdx.ui.components

import androidx.compose.ui.graphics.Color
import com.kimhs.liverdx.ml.Colormap

/** The colour ramps drawn under the two map panels, sampled from the same [Colormap] anchors. */
fun infernoRamp(): List<Color> = Colormap.INFERNO.map { Color(it) }

fun turboRamp(): List<Color> = Colormap.TURBO.map { Color(it) }
