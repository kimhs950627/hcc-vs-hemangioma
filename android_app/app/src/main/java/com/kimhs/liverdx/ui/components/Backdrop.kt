package com.kimhs.liverdx.ui.components

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.runtime.Composable
import com.kimhs.liverdx.ui.theme.Palette

/**
 * Opaque app backdrop: a single off-centre radial lift instead of a flat fill, so the surface has
 * depth without gradients that read as decoration.
 */
@Composable
fun Backdrop(content: @Composable BoxScope.() -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxSize()
            .drawBehind {
                drawRect(
                    Brush.radialGradient(
                        colors = listOf(Palette.Surface, Palette.Ink),
                        center = Offset(size.width * 0.5f, size.height * 0.22f),
                        radius = size.maxDimension * 0.85f,
                    )
                )
            },
        content = content,
    )
}
