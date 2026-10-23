package com.kimhs.liverdx.ui.components

import android.graphics.Bitmap
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithCache
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.style.TextAlign
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes

/**
 * The 1:1 image slot. Draws a dashed ruler frame while empty, the decoded frame once a photo
 * arrives, and dims it with a spinner while inference runs.
 */
@Composable
fun ImageSlot(
    preview: Bitmap?,
    running: Boolean,
    modifier: Modifier = Modifier,
) {
    Box(
        Modifier
            .fillMaxWidth()
            .aspectRatio(1f)
            .clip(Shapes.card)
            .background(Palette.Surface),
        contentAlignment = Alignment.Center,
    ) {
        if (preview == null) {
            Box(Modifier.fillMaxSize().dashedFrame())
            Column(
                Modifier.padding(Dimens.breathing),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(Dimens.tight),
            ) {
                Text("NO IMAGE LOADED", style = AppType.gaugeLabel, color = Palette.TextSecondary)
                Text(
                    text = "초음파 사진을 촬영하거나\n갤러리에서 선택하세요",
                    style = AppType.body,
                    color = Palette.TextMuted,
                    textAlign = TextAlign.Center,
                )
            }
        } else {
            val image = remember(preview) { preview.asImageBitmap() }
            Image(
                bitmap = image,
                contentDescription = "선택한 간초음파 영상",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize(),
            )
            if (running) {
                Box(
                    Modifier
                        .fillMaxSize()
                        .background(Palette.Ink.copy(alpha = 0.62f)),
                    contentAlignment = Alignment.Center,
                ) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(Dimens.tight),
                    ) {
                        CircularProgressIndicator(
                            color = Palette.Benign,
                            strokeWidth = Dimens.progressStroke,
                            modifier = Modifier.padding(Dimens.micro),
                        )
                        Text("INFERENCE RUNNING", style = AppType.gaugeLabel, color = Palette.TextSecondary)
                    }
                }
            }
        }
    }
}

private fun Modifier.dashedFrame(): Modifier = drawWithCache {
    val dash = PathEffect.dashPathEffect(floatArrayOf(Dimens.dashOn.toPx(), Dimens.dashOff.toPx()))
    val outline = Path().apply {
        addRoundRect(
            RoundRect(
                rect = Rect(0f, 0f, size.width, size.height),
                cornerRadius = CornerRadius(Dimens.cardRadius.toPx()),
            )
        )
    }
    onDrawBehind { drawPath(outline, color = Palette.Hairline, style = Stroke(width = Dimens.hairline.toPx(), pathEffect = dash)) }
}
