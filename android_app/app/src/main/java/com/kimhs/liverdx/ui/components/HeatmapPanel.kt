package com.kimhs.liverdx.ui.components

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes

/**
 * One square visualisation thumbnail: caption label, framed image, colour ramp and a one-line
 * reading of what the map shows.
 */
@Composable
fun HeatmapPanel(
    title: String,
    caption: String,
    bitmap: Bitmap,
    ramp: List<Color>?,
    onOpen: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val image = remember(bitmap) { bitmap.asImageBitmap() }
    Column(
        modifier = modifier.clickable(onClick = onOpen),
        verticalArrangement = Arrangement.spacedBy(Dimens.tight),
    ) {
        Text(text = title.uppercase(), style = AppType.gaugeLabel, color = Palette.TextSecondary)
        Box(
            Modifier
                .fillMaxWidth()
                .aspectRatio(1f)
                .clip(Shapes.card)
                .background(Palette.Surface)
                .border(Dimens.hairline, Palette.Hairline, Shapes.card)
        ) {
            Image(
                bitmap = image,
                contentDescription = title,
                // Fit, not Crop: a map must show the whole frame, and the frame is usually not square.
                contentScale = ContentScale.Fit,
                filterQuality = FilterQuality.Medium,
                modifier = Modifier.fillMaxWidth().aspectRatio(1f),
            )
        }
        if (ramp != null) {
            Box(
                Modifier
                    .fillMaxWidth()
                    .height(Dimens.rampHeight)
                    .background(Brush.horizontalGradient(ramp))
            )
        }
        Text(text = caption, style = AppType.gaugeNote, color = Palette.TextMuted)
    }
}
