package com.kimhs.liverdx.ui.theme

import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp

/**
 * Every colour, spacing and radius in the app. Components must not hardcode hex or dp values
 * (checked by `scripts/design-lint.py`).
 */
object Palette {
    val Ink = Color(0xFF070A0F)
    val Surface = Color(0xFF0F151D)
    val SurfaceHigh = Color(0xFF17202B)
    val Hairline = Color(0xFF243141)
    val TextPrimary = Color(0xFFEAF1F8)
    val TextSecondary = Color(0xFF93A6B8)
    val TextMuted = Color(0xFF5E7386)
    val Malignant = Color(0xFFFF3D5A)
    val Benign = Color(0xFF22D3A6)
    val Caution = Color(0xFFF5B544)
}

object Dimens {
    val gutter = 16.dp
    val tight = 8.dp
    val breathing = 24.dp
    val slot = 320.dp
    val trackHeight = 2.dp
    val tickHeight = 8.dp
    val rampHeight = 3.dp
    val accentBar = 4.dp
    val hairline = 1.dp
    val cardRadius = 6.dp
    val controlRadius = 4.dp
    val touchTarget = 48.dp
    val icon = 18.dp
    val progressStroke = 2.dp
    val dashOn = 9.dp
    val dashOff = 7.dp
    val controlVertical = 12.dp
    val shutter = 120.dp

    // Row-level rhythm inside the scoreboard and chips.
    val hairGap = 2.dp
    val labelGap = 3.dp
    val micro = 4.dp
    val chipVertical = 6.dp
    val rowVertical = 10.dp
    val rowVerticalLarge = 14.dp
}

object Shapes {
    val card = RoundedCornerShape(Dimens.cardRadius)
    val control = RoundedCornerShape(Dimens.controlRadius)
}
