package com.kimhs.liverdx.ui

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.kimhs.liverdx.ml.Analysis
import com.kimhs.liverdx.ml.Colormap
import com.kimhs.liverdx.ml.HeatmapRenderer
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext

/** Wires the camera screen, the gallery picker and the overlay rendering to [AnalyzeScreen]. */
@Composable
fun AnalyzeRoute(
    modifier: Modifier = Modifier,
    viewModel: AnalysisViewModel = viewModel(),
) {
    val context = LocalContext.current
    val state by viewModel.state.collectAsStateWithLifecycle()
    val model by viewModel.model.collectAsStateWithLifecycle()
    val settings by viewModel.settings.collectAsStateWithLifecycle()

    var cameraOpen by remember { mutableStateOf(false) }
    var openLayer by remember { mutableStateOf<HeatmapLayer?>(null) }

    val pickImage = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { picked ->
        if (picked != null) viewModel.analyze(picked)
    }

    val done = state as? AnalyzeState.Done
    val attention by overlayState(done, HeatmapLayer.ATTENTION) { analysis, preview ->
        renderMap(analysis.attentionMap, analysis.grid, preview, Colormap::inferno)
    }
    val prototype by overlayState(done, HeatmapLayer.PROTOTYPE) { analysis, preview ->
        renderMap(analysis.prototypeMap, analysis.grid, preview, Colormap::turbo)
    }

    // Matches docs/ux-flow.md: back returns to the empty slot instead of leaving the app.
    val canReturnToEmpty = !cameraOpen && openLayer == null &&
        (state is AnalyzeState.Done || state is AnalyzeState.Failed)
    BackHandler(enabled = canReturnToEmpty) { viewModel.reset() }

    if (cameraOpen) {
        CameraCaptureScreen(
            onCaptured = { file ->
                cameraOpen = false
                viewModel.analyzeFile(file)
            },
            onFailure = { message ->
                cameraOpen = false
                viewModel.reportError(message)
            },
            onDismiss = { cameraOpen = false },
        )
        return
    }

    AnalyzeScreen(
        state = state,
        model = model,
        settings = settings,
        overlays = Overlays(cleaned = done?.cleaned, attention = attention, prototype = prototype),
        onOpenLayer = { openLayer = it },
        onToggleStage = viewModel::setStage,
        onResizeMode = viewModel::setResizeMode,
        actions = ScreenActions(
            capture = { cameraOpen = true },
            pick = {
                pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
            },
            reset = viewModel::reset,
            retry = viewModel::retry,
        ),
        modifier = modifier,
    )

    val active = openLayer
    val attentionBitmap = attention
    val prototypeBitmap = prototype
    if (active != null && done != null && attentionBitmap != null && prototypeBitmap != null) {
        val layers = linkedMapOf(HeatmapLayer.ORIGINAL to done.original)
        done.cleaned?.let { layers[HeatmapLayer.CLEANED] = it }
        layers[HeatmapLayer.ATTENTION] = attentionBitmap
        layers[HeatmapLayer.PROTOTYPE] = prototypeBitmap
        ImageViewerDialog(
            layers = layers,
            startLayer = if (layers.containsKey(active)) active else HeatmapLayer.ORIGINAL,
            onDismiss = { openLayer = null },
        )
    }
}

/**
 * Renders one overlay off the main thread and swaps it in, recycling the bitmap it replaces.
 *
 * The upscale touches every pixel of the frame, so it must not run during composition.
 */
@Composable
private fun overlayState(
    done: AnalyzeState.Done?,
    layer: HeatmapLayer,
    render: (Analysis, Bitmap) -> Bitmap,
): State<Bitmap?> = produceState<Bitmap?>(initialValue = null, done, layer) {
    val previous = value
    val rendered = done?.let { withContext(Dispatchers.Default) { render(it.analysis, it.original) } }
    value = rendered
    if (previous != null && previous !== rendered) previous.recycle()
}

private fun renderMap(
    values: FloatArray,
    grid: Int,
    preview: Bitmap,
    colormap: (Float) -> Int,
): Bitmap = HeatmapRenderer.heatmapBitmap(values, grid, preview.width, preview.height, colormap)
