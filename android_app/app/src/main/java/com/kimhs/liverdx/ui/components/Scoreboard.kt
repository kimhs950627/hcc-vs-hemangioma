package com.kimhs.liverdx.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.IntrinsicSize
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.style.TextAlign
import com.kimhs.liverdx.ml.Analysis
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes
import java.util.Locale

/** One row of the scoreboard panel. */
data class MetricSpec(
    val label: String,
    val value: Float,
    val cutoff: Float?,
    val min: Float,
    val max: Float,
    val decimals: Int,
) {
    fun format(): String = String.format(Locale.US, "%.${decimals}f", value)
}

/** The headline verdict: label on the left, P(HCC) on the right, accent bar on the far left. */
@Composable
fun VerdictCard(analysis: Analysis, modifier: Modifier = Modifier) {
    val accent = if (analysis.isMalignant) Palette.Malignant else Palette.Benign
    Row(
        modifier = modifier
            .fillMaxWidth()
            .height(IntrinsicSize.Min)
            .clip(Shapes.card)
            .background(Palette.Surface)
            .border(Dimens.hairline, Palette.Hairline, Shapes.card),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Box(
            Modifier
                .width(Dimens.accentBar)
                .fillMaxHeight()
                .background(accent)
        )
        Column(
            Modifier
                .weight(1f)
                .padding(horizontal = Dimens.gutter, vertical = Dimens.rowVerticalLarge),
            verticalArrangement = Arrangement.spacedBy(Dimens.micro),
        ) {
            Text(
                text = if (analysis.isMalignant) "HCC" else "HEMANGIOMA",
                style = AppType.verdictLabel,
                color = accent,
            )
            Text(
                text = if (analysis.isMalignant) "악성 의심 · Malignant concern" else "양성 · Benign",
                style = AppType.verdictDetail,
                color = Palette.TextSecondary,
            )
            Text(
                text = if (analysis.unanimous) "3/3 markers agree" else "markers disagree — review overlays",
                style = AppType.gaugeNote,
                color = if (analysis.unanimous) Palette.TextMuted else Palette.Caution,
            )
        }
        Text(
            text = String.format(Locale.US, "%.1f%%", analysis.confidence * 100f),
            style = AppType.verdictPercent,
            color = accent,
            textAlign = TextAlign.End,
            modifier = Modifier.padding(end = Dimens.gutter),
        )
    }
}

/** One instrument row: label, tabular value, cutoff tick on a normalised track. */
@Composable
fun MetricRow(spec: MetricSpec, accent: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier.fillMaxWidth().padding(vertical = Dimens.rowVertical),
        verticalArrangement = Arrangement.spacedBy(Dimens.chipVertical),
    ) {
        Row(verticalAlignment = Alignment.Bottom) {
            Text(
                text = spec.label.uppercase(Locale.US),
                style = AppType.gaugeLabel,
                color = Palette.TextSecondary,
                modifier = Modifier.weight(1f).padding(bottom = Dimens.labelGap),
            )
            Text(text = spec.format(), style = AppType.gaugeValue, color = accent)
        }
        Track(spec = spec, accent = accent)
        CutoffNote(spec = spec)
    }
}

@Composable
private fun Track(spec: MetricSpec, accent: Color) {
    val span = spec.max - spec.min
    Box(
        Modifier.fillMaxWidth().height(Dimens.tickHeight),
        contentAlignment = Alignment.CenterStart,
    ) {
        // Rail, filled portion, then the cutoff tick, so the tick always stays visible on top.
        Box(
            Modifier
                .fillMaxWidth()
                .height(Dimens.trackHeight)
                .background(Palette.Hairline, RoundedCornerShape(Dimens.trackHeight))
        )
        Box(
            Modifier
                .fillMaxWidth(positionOf(spec.value, spec.min, span))
                .height(Dimens.trackHeight)
                .background(accent, RoundedCornerShape(Dimens.trackHeight))
        )
        spec.cutoff?.let { cutoff ->
            Box(
                Modifier
                    .fillMaxWidth(positionOf(cutoff, spec.min, span))
                    .height(Dimens.tickHeight),
                contentAlignment = Alignment.CenterStart,
            ) {
                Box(
                    Modifier
                        .width(Dimens.hairline)
                        .height(Dimens.tickHeight)
                        .background(Palette.Caution)
                )
            }
        }
    }
}

@Composable
private fun CutoffNote(spec: MetricSpec) {
    val cutoff = spec.cutoff ?: return
    val above = spec.value >= cutoff
    Text(
        text = String.format(
            Locale.US,
            "%s cutoff %.${spec.decimals}f",
            if (above) "ABOVE" else "BELOW",
            cutoff,
        ),
        style = AppType.gaugeNote,
        color = if (above) Palette.Caution else Palette.TextMuted,
    )
}

private fun positionOf(value: Float, min: Float, span: Float): Float =
    ((value - min) / span).coerceIn(0f, 1f)

/** The four paper markers, in the order the report prints them. */
fun metricsOf(analysis: Analysis): List<Pair<MetricSpec, Color>> = listOf(
    MetricSpec(
        label = "Confidence · P(HCC)",
        value = analysis.confidence,
        cutoff = analysis.cutoffs.confidence,
        min = 0f,
        max = 1f,
        decimals = 4,
    ) to verdictColor(analysis.isMalignant),
    MetricSpec(
        label = "HCC Cosine Score",
        value = analysis.hccCosine,
        cutoff = analysis.cutoffs.hccCosine,
        min = -1f,
        max = 1f,
        decimals = 3,
    ) to verdictColor(analysis.byHccCosine),
    MetricSpec(
        label = "Hemangioma Cosine Score",
        value = analysis.hemCosine,
        cutoff = null,
        min = -1f,
        max = 1f,
        decimals = 3,
    ) to Palette.TextPrimary,
    MetricSpec(
        label = "Δscore (margin)",
        value = analysis.delta,
        cutoff = analysis.cutoffs.delta,
        min = -2f,
        max = 2f,
        decimals = 3,
    ) to verdictColor(analysis.byDelta),
)

fun verdictColor(malignant: Boolean): Color = if (malignant) Palette.Malignant else Palette.Benign

/** Thin divider used between scoreboard rows. */
@Composable
fun Hairline(modifier: Modifier = Modifier) {
    Spacer(
        modifier
            .fillMaxWidth()
            .height(Dimens.hairline)
            .background(Palette.Hairline)
    )
}
