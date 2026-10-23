package com.kimhs.liverdx.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Scheme = darkColorScheme(
    primary = Palette.TextPrimary,
    onPrimary = Palette.Ink,
    secondary = Palette.Benign,
    onSecondary = Palette.Ink,
    tertiary = Palette.Malignant,
    background = Palette.Ink,
    onBackground = Palette.TextPrimary,
    surface = Palette.Surface,
    onSurface = Palette.TextPrimary,
    surfaceVariant = Palette.SurfaceHigh,
    onSurfaceVariant = Palette.TextSecondary,
    outline = Palette.Hairline,
    error = Palette.Malignant,
    onError = Color.White,
)

/** Dark-only: the app is read in a darkened reading room next to the ultrasound console. */
@Composable
fun LiverDxTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = Scheme,
        typography = AppTypography,
        content = content,
    )
}
