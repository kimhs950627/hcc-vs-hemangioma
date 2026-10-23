package com.kimhs.liverdx.ui.theme

import androidx.compose.material3.Typography
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.Font
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.em
import androidx.compose.ui.unit.sp
import com.kimhs.liverdx.R

/**
 * Space Grotesk (OFL-1.1) — an instrument-style grotesque with flat terminals. The Latin subset is
 * bundled; Korean glyphs fall back to the platform font, which is intentional.
 */
val SpaceGrotesk = FontFamily(
    Font(R.font.space_grotesk_400, FontWeight.Normal),
    Font(R.font.space_grotesk_500, FontWeight.Medium),
    Font(R.font.space_grotesk_700, FontWeight.Bold),
)

/** Screen-specific styles. Tabular numerals keep score columns aligned. */
object AppType {
    val screenTitle = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = 21.sp, letterSpacing = (-0.01).em)
    val screenSubtitle = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Normal, fontSize = 11.sp, letterSpacing = 0.08.em)

    val verdictLabel = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = 26.sp, letterSpacing = (-0.01).em)
    val verdictDetail = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Normal, fontSize = 12.sp, letterSpacing = 0.02.em)
    val verdictPercent = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = 30.sp, fontFeatureSettings = "tnum")

    val gaugeLabel = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Medium, fontSize = 11.sp, letterSpacing = 0.1.em)
    val gaugeValue = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Bold, fontSize = 20.sp, fontFeatureSettings = "tnum")
    val gaugeNote = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Normal, fontSize = 10.sp, letterSpacing = 0.04.em)

    val buttonLabel = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Medium, fontSize = 14.sp, letterSpacing = 0.06.em)
    val chip = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Medium, fontSize = 10.sp, letterSpacing = 0.06.em)
    val body = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Normal, fontSize = 13.sp, lineHeight = 19.sp)
    val disclaimer = TextStyle(fontFamily = SpaceGrotesk, fontWeight = FontWeight.Normal, fontSize = 10.sp, lineHeight = 15.sp, letterSpacing = 0.02.em)
}

val AppTypography = Typography(
    headlineSmall = AppType.screenTitle,
    titleLarge = AppType.verdictLabel,
    bodyMedium = AppType.body,
    labelSmall = AppType.chip,
)
