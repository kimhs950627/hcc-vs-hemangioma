package com.kimhs.liverdx.ui

import android.graphics.Bitmap
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.kimhs.liverdx.R
import com.kimhs.liverdx.ml.ResizeMode
import com.kimhs.liverdx.ui.components.Backdrop
import com.kimhs.liverdx.ui.components.Hairline
import com.kimhs.liverdx.ui.components.HeatmapPanel
import com.kimhs.liverdx.ui.components.ImageSlot
import com.kimhs.liverdx.ui.components.MetricRow
import com.kimhs.liverdx.ui.components.PipelinePanel
import com.kimhs.liverdx.ui.components.VerdictCard
import com.kimhs.liverdx.ui.components.clickableNoRipple
import com.kimhs.liverdx.ui.components.infernoRamp
import com.kimhs.liverdx.ui.components.metricsOf
import com.kimhs.liverdx.ui.components.turboRamp
import com.kimhs.liverdx.ui.theme.AppType
import com.kimhs.liverdx.ui.theme.Dimens
import com.kimhs.liverdx.ui.theme.Palette
import com.kimhs.liverdx.ui.theme.Shapes
import kotlinx.coroutines.delay
import java.util.Locale

/** Callbacks the screen needs; keeps the composable free of launcher plumbing. */
data class ScreenActions(
    val capture: () -> Unit,
    val pick: () -> Unit,
    val reset: () -> Unit,
    val retry: () -> Unit,
)

/** Model identity, shown so any screenshot can be traced back to an exact flatbuffer. */
data class ModelSummary(
    val training: String,
    val precision: String,
    val cosineDrift: Float,
    val embedDim: Int,
    val patchGrid: Int,
    val inputSize: Int,
    val sha256Prefix: String,
)

/** Derived images the panels and the viewer need. */
data class Overlays(
    val cleaned: Bitmap?,
    val attention: Bitmap?,
    val prototype: Bitmap?,
)

@Composable
fun AnalyzeScreen(
    state: AnalyzeState,
    model: ModelSummary?,
    settings: Settings,
    overlays: Overlays,
    onOpenLayer: (HeatmapLayer) -> Unit,
    onToggleStage: (String, Boolean) -> Unit,
    onResizeMode: (ResizeMode) -> Unit,
    actions: ScreenActions,
    modifier: Modifier = Modifier,
) {
    Backdrop {
        Column(
            modifier = modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .windowInsetsPadding(WindowInsets.safeDrawing)
                .padding(horizontal = Dimens.gutter, vertical = Dimens.tight),
            verticalArrangement = Arrangement.spacedBy(Dimens.gutter),
        ) {
            Header(model)

            when (state) {
                AnalyzeState.Booting -> BootNote("모델을 불러오는 중…")
                is AnalyzeState.ModelUnavailable -> ErrorCard(
                    title = "MODEL UNAVAILABLE",
                    message = state.message,
                    actions = null,
                )

                AnalyzeState.Ready -> {
                    ImageSlot(preview = null, running = false)
                    ActionRow(actions)
                }

                is AnalyzeState.Running -> ImageSlot(preview = state.original, running = true)

                is AnalyzeState.Done -> {
                    ResultSection(state, settings, overlays, onOpenLayer, onToggleStage, onResizeMode)
                    ActionRow(actions)
                }

                is AnalyzeState.Failed -> {
                    ImageSlot(preview = state.original, running = false)
                    ActionRow(actions)
                    ErrorCard("ANALYSIS FAILED", state.message, actions)
                }
            }

            Disclaimer()
            Spacer(Modifier.height(Dimens.breathing))
        }
    }
}

@Composable
private fun Header(model: ModelSummary?) {
    Row(verticalAlignment = Alignment.Top) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text("LiverDx", style = AppType.screenTitle, color = Palette.TextPrimary)
            Text("HCC vs HEMANGIOMA · B-MODE US", style = AppType.screenSubtitle, color = Palette.TextMuted)
        }
        model?.let { ModelChip(it) }
    }
}

@Composable
private fun ModelChip(model: ModelSummary) {
    Column(
        Modifier
            .clip(Shapes.control)
            .background(Palette.Surface)
            .border(Dimens.hairline, Palette.Hairline, Shapes.control)
            .padding(horizontal = Dimens.tight, vertical = Dimens.chipVertical),
        horizontalAlignment = Alignment.End,
        verticalArrangement = Arrangement.spacedBy(Dimens.hairGap),
    ) {
        Text(model.training, style = AppType.chip, color = Palette.TextSecondary)
        Text(
            String.format(
                Locale.US,
                "%s · Δcos ≤ %.1e",
                model.precision,
                model.cosineDrift,
            ),
            style = AppType.chip,
            color = if (model.precision == "fp32") Palette.TextMuted else Palette.Caution,
        )
        Text(
            "${model.embedDim}-d · ${model.patchGrid}×${model.patchGrid} · ${model.inputSize}²",
            style = AppType.chip,
            color = Palette.TextMuted,
        )
        Text("sha256 ${model.sha256Prefix}", style = AppType.chip, color = Palette.TextMuted)
    }
}

@Composable
private fun ActionRow(actions: ScreenActions) {
    Row(horizontalArrangement = Arrangement.spacedBy(Dimens.tight)) {
        PrimaryAction(
            label = "촬영",
            iconRes = R.drawable.ic_camera,
            onClick = actions.capture,
            modifier = Modifier.weight(1f),
        )
        SecondaryAction(
            label = "갤러리",
            iconRes = R.drawable.ic_gallery,
            onClick = actions.pick,
            modifier = Modifier.weight(1f),
        )
        SecondaryAction(
            label = "초기화",
            iconRes = R.drawable.ic_reset,
            onClick = actions.reset,
            modifier = Modifier.weight(1f),
        )
    }
}

@Composable
private fun PrimaryAction(label: String, iconRes: Int, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .height(Dimens.touchTarget)
            .clip(Shapes.control)
            .background(Palette.Benign.copy(alpha = 0.14f))
            .border(Dimens.hairline, Palette.Benign.copy(alpha = 0.55f), Shapes.control)
            .clickableNoRipple(onClick),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(painterResource(iconRes), contentDescription = null, tint = Palette.Benign, modifier = Modifier.size(Dimens.icon))
        Spacer(Modifier.width(Dimens.tight))
        Text(label, style = AppType.buttonLabel, color = Palette.Benign)
    }
}

@Composable
private fun SecondaryAction(label: String, iconRes: Int, onClick: () -> Unit, modifier: Modifier = Modifier) {
    Row(
        modifier = modifier
            .height(Dimens.touchTarget)
            .clip(Shapes.control)
            .background(Palette.Surface)
            .border(Dimens.hairline, Palette.Hairline, Shapes.control)
            .clickableNoRipple(onClick),
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Icon(painterResource(iconRes), contentDescription = null, tint = Palette.TextSecondary, modifier = Modifier.size(Dimens.icon))
        Spacer(Modifier.width(Dimens.tight))
        Text(label, style = AppType.buttonLabel, color = Palette.TextSecondary)
    }
}

@Composable
private fun ResultSection(
    state: AnalyzeState.Done,
    settings: Settings,
    overlays: Overlays,
    onOpenLayer: (HeatmapLayer) -> Unit,
    onToggleStage: (String, Boolean) -> Unit,
    onResizeMode: (ResizeMode) -> Unit,
) {
    val analysis = state.analysis
    val inputFrame = overlays.cleaned ?: state.original

    Reveal(0) { VerdictCard(analysis) }

    Column(Modifier.fillMaxWidth()) {
        metricsOf(analysis).forEachIndexed { index, (spec, accent) ->
            if (index > 0) Hairline()
            Reveal(index + 1) { MetricRow(spec, accent) }
        }
    }

    Reveal(STAGE_REVEAL) {
        PipelinePanel(
            settings = settings,
            report = state.pipeline,
            onToggleStage = onToggleStage,
            onResizeMode = onResizeMode,
        )
    }

    Reveal(STAGE_REVEAL + 1) {
        Column(verticalArrangement = Arrangement.spacedBy(Dimens.tight)) {
            Text("VISUALISATION · 탭하면 확대", style = AppType.gaugeLabel, color = Palette.TextSecondary)
            Row(horizontalArrangement = Arrangement.spacedBy(Dimens.tight)) {
                HeatmapPanel(
                    title = if (overlays.cleaned != null) "Cleaned frame" else "Raw frame",
                    caption = String.format(
                        Locale.US,
                        "%d×%d · %s",
                        state.pipeline.sourceWidth,
                        state.pipeline.sourceHeight,
                        if (overlays.cleaned != null) "${state.pipeline.appliedStages.size} stages" else "no cleanup",
                    ),
                    bitmap = inputFrame,
                    ramp = null,
                    onOpen = { onOpenLayer(HeatmapLayer.ORIGINAL) },
                    modifier = Modifier.weight(1f),
                )
                overlays.attention?.let {
                    HeatmapPanel(
                        title = "CA attention",
                        caption = "rollout, 4 layers",
                        bitmap = it,
                        ramp = infernoRamp(),
                        onOpen = { onOpenLayer(HeatmapLayer.ATTENTION) },
                        modifier = Modifier.weight(1f),
                    )
                }
                overlays.prototype?.let {
                    HeatmapPanel(
                        title = "HCC prototype",
                        caption = "per-patch cosine, k=4",
                        bitmap = it,
                        ramp = turboRamp(),
                        onOpen = { onOpenLayer(HeatmapLayer.PROTOTYPE) },
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }
    }

    Reveal(STAGE_REVEAL + 2) {
        Text(
            text = String.format(
                Locale.US,
                "clean %d ms · inference %d ms · CPU · %s · %d×%d patch grid",
                state.pipeline.processMillis,
                analysis.inferenceMillis,
                settings.resizeMode.label,
                analysis.grid,
                analysis.grid,
            ),
            style = AppType.gaugeNote,
            color = Palette.TextMuted,
        )
    }
}

@Composable
private fun ErrorCard(title: String, message: String, actions: ScreenActions?) {
    Column(
        Modifier
            .fillMaxWidth()
            .clip(Shapes.card)
            .background(Palette.Surface)
            .border(Dimens.hairline, Palette.Malignant.copy(alpha = 0.5f), Shapes.card)
            .padding(Dimens.gutter),
        verticalArrangement = Arrangement.spacedBy(Dimens.tight),
    ) {
        Text(title, style = AppType.gaugeLabel, color = Palette.Malignant)
        Text(message, style = AppType.body, color = Palette.TextSecondary)
        if (actions != null) {
            Row(
                Modifier
                    .height(Dimens.touchTarget)
                    .clip(Shapes.control)
                    .border(Dimens.hairline, Palette.Hairline, Shapes.control)
                    .clickableNoRipple(actions.retry)
                    .padding(horizontal = Dimens.gutter),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("다시 시도", style = AppType.buttonLabel, color = Palette.TextPrimary)
            }
        }
    }
}

@Composable
private fun BootNote(text: String) {
    Text(text, style = AppType.body, color = Palette.TextMuted)
}

@Composable
private fun Disclaimer() {
    Box(
        Modifier
            .fillMaxWidth()
            .clip(Shapes.card)
            .background(Palette.Surface.copy(alpha = 0.6f))
            .padding(Dimens.gutter)
    ) {
        Text(
            text = "연구·디버깅용 도구입니다. 진단 또는 치료 결정에 사용할 수 없습니다. " +
                "Not a medical device — research and debugging use only. " +
                "촬영본은 기기 밖으로 나가지 않으며, 디코드 직후 캐시에서 삭제됩니다. " +
                "출력값은 SMC-LUD validation cohort 기준 cut-off 와 비교한 값입니다.",
            style = AppType.disclaimer,
            color = Palette.TextMuted,
            textAlign = TextAlign.Start,
        )
    }
}

/** One-shot staggered entrance: each element fades and lifts 40 ms after the previous one. */
@Composable
private fun Reveal(index: Int, content: @Composable () -> Unit) {
    var shown by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) {
        delay(index * REVEAL_STEP_MILLIS)
        shown = true
    }
    AnimatedVisibility(
        visible = shown,
        enter = fadeIn(tween(REVEAL_DURATION_MILLIS, easing = FastOutSlowInEasing)) +
            slideInVertically(tween(REVEAL_DURATION_MILLIS, easing = FastOutSlowInEasing)) { it / 6 },
    ) {
        content()
    }
}

private const val STAGE_REVEAL = 5
private const val REVEAL_STEP_MILLIS = 40L
private const val REVEAL_DURATION_MILLIS = 220
