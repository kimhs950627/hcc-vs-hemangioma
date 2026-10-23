package com.kimhs.liverdx.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.material3.Icon
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.kimhs.liverdx.R
import com.kimhs.liverdx.ml.HeatmapRenderer
import com.kimhs.liverdx.ui.components.clickableNoRipple
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.util.Locale

/**
 * Full-screen viewer with an opacity slider. Blending runs off the main thread because it touches
 * every pixel of a 384x384 frame.
 */
@Composable
fun ImageViewerDialog(
    layers: Map<HeatmapLayer, Bitmap>,
    startLayer: HeatmapLayer,
    onDismiss: () -> Unit,
) {
    var selected by remember(layers) {
        mutableStateOf(if (layers.containsKey(startLayer)) startLayer else HeatmapLayer.ORIGINAL)
    }
    var opacity by remember { mutableFloatStateOf(DEFAULT_OPACITY) }

    val original = layers[HeatmapLayer.ORIGINAL]
    val overlay = layers[selected]
    val shown by produceState(initialValue = original, selected, opacity) {
        val previous = value
        val rendered = if (overlay == null || original == null || overlay === original) {
            original
        } else {
            withContext(Dispatchers.Default) { HeatmapRenderer.blend(original, overlay, opacity) }
        }
        value = rendered
        // Only blended frames are owned here; `original` and `overlay` belong to the screen.
        if (previous != null && previous !== rendered && previous !== original && previous !== overlay) {
            previous.recycle()
        }
    }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Column(
            Modifier
                .fillMaxSize()
                .background(Palette.Ink)
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(Dimens.gutter),
            verticalArrangement = Arrangement.spacedBy(Dimens.gutter),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(Dimens.hairGap)) {
                    Text(selected.title, style = AppType.gaugeLabel, color = Palette.TextPrimary)
                    Text(selected.note, style = AppType.gaugeNote, color = Palette.TextMuted)
                }
                Icon(
                    painter = painterResource(R.drawable.ic_close),
                    contentDescription = "닫기",
                    tint = Palette.TextSecondary,
                    modifier = Modifier
                        .size(Dimens.touchTarget)
                        .clip(Shapes.control)
                        .clickableNoRipple(onDismiss)
                        .padding(Dimens.rowVerticalLarge),
                )
            }

            Box(
                Modifier
                    .fillMaxWidth()
                    .aspectRatio(1f)
                    .clip(Shapes.card)
                    .background(Palette.Surface)
                    .border(Dimens.hairline, Palette.Hairline, Shapes.card),
                contentAlignment = Alignment.Center,
            ) {
                shown?.let {
                    Image(
                        bitmap = it.asImageBitmap(),
                        contentDescription = selected.title,
                        modifier = Modifier.fillMaxSize(),
                    )
                }
            }

            Row(horizontalArrangement = Arrangement.spacedBy(Dimens.tight)) {
                layers.keys.forEach { layer ->
                    LayerChip(
                        layer = layer,
                        selected = layer == selected,
                        onClick = { selected = layer },
                        modifier = Modifier.weight(1f),
                    )
                }
            }

            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("OVERLAY", style = AppType.gaugeLabel, color = Palette.TextSecondary)
                Slider(
                    value = opacity,
                    onValueChange = { opacity = it },
                    enabled = selected != HeatmapLayer.ORIGINAL,
                    colors = SliderDefaults.colors(
                        thumbColor = Palette.Benign,
                        activeTrackColor = Palette.Benign,
                        inactiveTrackColor = Palette.Hairline,
                    ),
                    modifier = Modifier.weight(1f).padding(horizontal = Dimens.tight),
                )
                Text(
                    text = String.format(Locale.US, "%3.0f%%", opacity * 100f),
                    style = AppType.gaugeValue,
                    color = Palette.TextPrimary,
                )
            }
        }
    }
}

@Composable
private fun LayerChip(
    layer: HeatmapLayer,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Box(
        modifier
            .height(Dimens.touchTarget)
            .clip(Shapes.control)
            .background(if (selected) Palette.SurfaceHigh else Palette.Surface)
            .border(
                Dimens.hairline,
                if (selected) Palette.Benign else Palette.Hairline,
                Shapes.control,
            )
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            text = layer.chip,
            style = AppType.chip,
            color = if (selected) Palette.Benign else Palette.TextSecondary,
        )
    }
}

private const val DEFAULT_OPACITY = 0.45f
