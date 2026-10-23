package com.kimhs.liverdx.ui.components

import androidx.compose.foundation.clickable
import androidx.compose.ui.Modifier

/** Clickable without the material ripple: the surfaces here are instrument panels, not buttons. */
fun Modifier.clickableNoRipple(onClick: () -> Unit): Modifier =
    clickable(interactionSource = null, indication = null, onClick = onClick)
