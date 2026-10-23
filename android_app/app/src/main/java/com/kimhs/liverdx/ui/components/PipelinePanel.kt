package com.kimhs.liverdx.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.kimhs.liverdx.ml.PhotoPreprocess
import com.kimhs.liverdx.ml.ResizeMode
import com.kimhs.liverdx.ui.PipelineReport
import com.kimhs.liverdx.ui.Settings
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes
import java.util.Locale

/** Stage ids from the model metadata, with the labels this screen shows. */
private val STAGE_LABELS = linkedMapOf(
    PhotoPreprocess.STAGE_GLARE to ("반사광 제거" to "remove_glare"),
    PhotoPreprocess.STAGE_FLATTEN to ("조명 평탄화" to "flatten_illumination"),
    PhotoPreprocess.STAGE_DESATURATE to ("그레이스케일 정렬" to "desaturate"),
    PhotoPreprocess.STAGE_DENOISE to ("노이즈 제거" to "denoise"),
    PhotoPreprocess.STAGE_CONTRAST to ("대비 정규화" to "stretch_contrast"),
)

/**
 * The cleanup switchboard. Every stage can be flipped while a frame is on screen and the analysis
 * re-runs, so the effect of each one is directly visible in the panels above.
 */
@Composable
fun PipelinePanel(
    settings: Settings,
    report: PipelineReport,
    onToggleStage: (String, Boolean) -> Unit,
    onResizeMode: (ResizeMode) -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(Shapes.card)
            .background(Palette.Surface)
            .border(Dimens.hairline, Palette.Hairline, Shapes.card)
            .padding(Dimens.gutter),
        verticalArrangement = Arrangement.spacedBy(Dimens.tight),
    ) {
        Row {
            Text(
                "PHOTO CLEANUP",
                style = AppType.gaugeLabel,
                color = Palette.TextSecondary,
                modifier = Modifier.weight(1f),
            )
            Text(
                String.format(
                    Locale.US,
                    "%d/%d · glare %.1f%%",
                    report.appliedStages.size,
                    STAGE_LABELS.size,
                    report.glareFraction * 100f,
                ),
                style = AppType.gaugeNote,
                color = Palette.TextMuted,
            )
        }

        STAGE_LABELS.forEach { (stage, labels) ->
            StageRow(
                primary = labels.first,
                secondary = labels.second,
                checked = settings.options.isEnabled(stage),
                onCheckedChange = { onToggleStage(stage, it) },
            )
        }

        Hairline(Modifier.padding(vertical = Dimens.chipVertical))

        Text("RESAMPLE", style = AppType.gaugeLabel, color = Palette.TextSecondary)
        Row(horizontalArrangement = Arrangement.spacedBy(Dimens.tight)) {
            ResizeMode.entries.forEach { mode ->
                ResizeChip(
                    mode = mode,
                    selected = mode == settings.resizeMode,
                    onClick = { onResizeMode(mode) },
                    modifier = Modifier.weight(1f),
                )
            }
        }
        Text(
            text = String.format(
                Locale.US,
                "source %d×%d → 384×384 · %s",
                report.sourceWidth,
                report.sourceHeight,
                settings.resizeMode.source,
            ),
            style = AppType.gaugeNote,
            color = Palette.TextMuted,
        )
    }
}

@Composable
private fun StageRow(
    primary: String,
    secondary: String,
    checked: Boolean,
    onCheckedChange: (Boolean) -> Unit,
) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(primary, style = AppType.body, color = Palette.TextPrimary, modifier = Modifier.weight(1f))
        Text(secondary, style = AppType.gaugeNote, color = Palette.TextMuted)
        Spacer(Modifier.width(Dimens.tight))
        Switch(
            checked = checked,
            onCheckedChange = onCheckedChange,
            colors = SwitchDefaults.colors(
                checkedThumbColor = Palette.Ink,
                checkedTrackColor = Palette.Benign,
                uncheckedThumbColor = Palette.TextMuted,
                uncheckedTrackColor = Palette.SurfaceHigh,
                uncheckedBorderColor = Palette.Hairline,
            ),
        )
    }
}

@Composable
private fun ResizeChip(
    mode: ResizeMode,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier
            .clip(Shapes.control)
            .background(if (selected) Palette.SurfaceHigh else Palette.Surface)
            .border(
                Dimens.hairline,
                if (selected) Palette.Benign else Palette.Hairline,
                Shapes.control,
            )
            .clickableNoRipple(onClick)
            .padding(horizontal = Dimens.tight, vertical = Dimens.tight),
        verticalArrangement = Arrangement.spacedBy(Dimens.hairGap),
    ) {
        Text(
            text = mode.label.uppercase(Locale.US),
            style = AppType.chip,
            color = if (selected) Palette.Benign else Palette.TextSecondary,
            textAlign = TextAlign.Start,
        )
        Text(mode.source, style = AppType.gaugeNote, color = Palette.TextMuted)
    }
}
