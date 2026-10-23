package com.kimhs.liverdx.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.kimhs.liverdx.ml.Analysis
import com.kimhs.liverdx.ml.Classifier
import com.kimhs.liverdx.ml.FrameProcessor
import com.kimhs.liverdx.ml.ModelBundle
import com.kimhs.liverdx.ml.PhotoPreprocess
import com.kimhs.liverdx.ml.ResizeMode
import com.kimhs.liverdx.util.CaptureFiles
import com.kimhs.liverdx.util.ImageLoading
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/** What the pipeline did to the frame, so the UI can show and reproduce it. */
data class PipelineReport(
    val resizeMode: ResizeMode,
    val appliedStages: List<String>,
    val glareFraction: Float,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val processMillis: Long,
    val inferenceMillis: Long,
)

/** Screen state machine. `original` is null only when decoding itself failed. */
sealed interface AnalyzeState {
    data object Booting : AnalyzeState
    data class ModelUnavailable(val message: String) : AnalyzeState
    data object Ready : AnalyzeState
    data class Running(val original: Bitmap) : AnalyzeState
    data class Done(
        val original: Bitmap,
        val cleaned: Bitmap?,
        val analysis: Analysis,
        val pipeline: PipelineReport,
    ) : AnalyzeState
    data class Failed(val original: Bitmap?, val message: String) : AnalyzeState
}

/** Cleanup toggles plus the resampler choice, both live-editable while a frame is on screen. */
data class Settings(
    val options: PhotoPreprocess.Options = PhotoPreprocess.Options(),
    val resizeMode: ResizeMode = ResizeMode.BILINEAR_UINT8,
)

/**
 * Owns the interpreter (built once on a background thread), the decode step, the photo cleanup and
 * the inference step. The view layer only renders [state] and edits [settings].
 *
 * Two invariants keep the native interpreter and the bitmaps safe:
 *  - only one run is in flight ([generation] fences stale coroutines, [inferenceLock] serialises the
 *    native `run` call),
 *  - a bitmap is recycled only once it has been replaced in [state].
 */
class AnalysisViewModel(application: Application) : AndroidViewModel(application) {

    private val _state = MutableStateFlow<AnalyzeState>(AnalyzeState.Booting)
    val state: StateFlow<AnalyzeState> = _state.asStateFlow()

    private val _model = MutableStateFlow<ModelSummary?>(null)
    val model: StateFlow<ModelSummary?> = _model.asStateFlow()

    private val _settings = MutableStateFlow(Settings())
    val settings: StateFlow<Settings> = _settings.asStateFlow()

    private val inferenceLock = Mutex()
    private var classifier: Classifier? = null
    private var processor: FrameProcessor? = null
    private var modelInputSize = DEFAULT_INPUT_SIZE
    private var lastUri: Uri? = null
    private var lastFile: File? = null
    private var sourceFrame: Bitmap? = null
    private var analysisJob: Job? = null
    private var generation = 0

    /** Set in [onCleared]; a fenced coroutine must not touch the interpreter after this. */
    @Volatile
    private var closed = false

    init {
        viewModelScope.launch {
            val loaded = withContext(Dispatchers.IO) {
                runCatching { ModelBundle.load(getApplication<Application>()) }
            }
            loaded
                .onSuccess { bundle ->
                    // Creating the Interpreter parses the 27 MB flatbuffer, so it stays off the main thread.
                    classifier = withContext(Dispatchers.IO) { Classifier(bundle) }
                    processor = FrameProcessor(bundle.meta.inputWidth, bundle.meta.inputHeight)
                    modelInputSize = bundle.meta.inputWidth
                    _model.value = ModelSummary(
                        training = bundle.meta.training,
                        precision = bundle.meta.precision,
                        cosineDrift = bundle.meta.cosineDrift,
                        embedDim = bundle.meta.embedDim,
                        patchGrid = bundle.meta.patchGrid,
                        inputSize = bundle.meta.inputWidth,
                        sha256Prefix = bundle.modelSha256.take(SHA_PREFIX_LENGTH),
                    )
                    _state.value = AnalyzeState.Ready
                }
                .onFailure { error ->
                    _state.value = AnalyzeState.ModelUnavailable(describe(error))
                }
        }
    }

    /** Decodes [uri] and runs the whole chain; supersedes any run already in flight. */
    fun analyze(uri: Uri) {
        lastUri = uri
        lastFile = null
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch {
            val decoded = withContext(Dispatchers.IO) {
                runCatching { ImageLoading.loadFromUri(getApplication<Application>(), uri, modelInputSize) }
            }
            decoded
                .onSuccess { runPipeline(it) }
                .onFailure { _state.value = AnalyzeState.Failed(currentOriginal(), describe(it)) }
        }
    }

    /** Decodes a camera file and runs the whole chain. */
    fun analyzeFile(file: File) {
        lastFile = file
        lastUri = null
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch {
            val decoded = withContext(Dispatchers.IO) {
                runCatching { ImageLoading.loadFromFile(file, modelInputSize) }
            }
            // The frame lives in memory from here on; the capture file does not need to survive.
            withContext(Dispatchers.IO) { CaptureFiles.delete(getApplication()) }
            decoded
                .onSuccess { runPipeline(it) }
                .onFailure { _state.value = AnalyzeState.Failed(currentOriginal(), describe(it)) }
        }
    }

    /** Re-runs the pipeline on the retained frame, without touching the camera or the gallery. */
    private suspend fun runPipeline(decoded: Bitmap) {
        val active = classifier ?: return
        val activeProcessor = processor ?: return

        val replacedSource = sourceFrame
        val replacedCleaned = cleanedFrame()
        val mine = ++generation
        sourceFrame = decoded
        if (replacedSource !== null && replacedSource !== decoded) replacedSource.recycle()
        _state.value = AnalyzeState.Running(decoded)
        if (replacedCleaned !== null) replacedCleaned.recycle()

        val run = _settings.value
        val outcome = withContext(Dispatchers.Default) {
            inferenceLock.withLock {
                if (closed) {
                    null
                } else {
                    runCatching {
                        val started = SystemClock.elapsedRealtime()
                        val processed = activeProcessor.process(decoded, run.options, run.resizeMode)
                        val analysis = active.analyze(processed.modelInput)
                        Triple(processed, analysis, SystemClock.elapsedRealtime() - started)
                    }
                }
            }
        }
        if (outcome == null || mine != generation) return

        outcome
            .onSuccess { (processed, analysis, elapsed) ->
                val cleaned = if (processed.cleanedApplied) {
                    withContext(Dispatchers.Default) {
                        FrameProcessor.toBitmap(processed.cleaned, processed.sourceWidth, processed.sourceHeight)
                    }
                } else {
                    null
                }
                if (mine != generation) {
                    cleaned?.recycle()
                    return
                }
                _state.value = AnalyzeState.Done(
                    original = decoded,
                    cleaned = cleaned,
                    analysis = analysis,
                    pipeline = PipelineReport(
                        resizeMode = run.resizeMode,
                        appliedStages = processed.appliedStages,
                        glareFraction = processed.glareFraction,
                        sourceWidth = processed.sourceWidth,
                        sourceHeight = processed.sourceHeight,
                        processMillis = elapsed - analysis.inferenceMillis,
                        inferenceMillis = analysis.inferenceMillis,
                    ),
                )
            }
            .onFailure { _state.value = AnalyzeState.Failed(decoded, describe(it)) }
    }

    /** Toggles one cleanup stage and re-runs on the retained frame. */
    fun setStage(stage: String, enabled: Boolean) {
        _settings.value = _settings.value.copy(options = _settings.value.options.withStage(stage, enabled))
        rerun()
    }

    fun setResizeMode(mode: ResizeMode) {
        _settings.value = _settings.value.copy(resizeMode = mode)
        rerun()
    }

    private fun rerun() {
        val frame = sourceFrame ?: return
        analysisJob?.cancel()
        analysisJob = viewModelScope.launch { runPipeline(frame) }
    }

    fun retry() {
        lastUri?.let(::analyze)
        lastFile?.let(::analyzeFile)
    }

    /** Reports a failure that happened before any frame existed (e.g. permission denied). */
    fun reportError(message: String) {
        _state.value = AnalyzeState.Failed(currentOriginal(), message)
    }

    fun reset() {
        generation++
        analysisJob?.cancel()
        analysisJob = null
        val original = currentOriginal()
        val cleaned = cleanedFrame()
        lastUri = null
        lastFile = null
        sourceFrame = null
        _state.value = AnalyzeState.Ready
        if (original != null && original !== cleaned) original.recycle()
        cleaned?.recycle()
    }

    override fun onCleared() {
        closed = true
        generation++
        analysisJob?.cancel()
        analysisJob = null
        val active = classifier
        classifier = null
        processor = null
        // Close only when no native run holds the lock. Otherwise the interpreter is released by GC:
        // closing it under a running inference would be a native use-after-free.
        if (active != null && inferenceLock.tryLock()) {
            try {
                active.close()
            } finally {
                inferenceLock.unlock()
            }
        }
        cleanedFrame()?.recycle()
        sourceFrame?.takeIf { it !== cleanedFrame() }?.recycle()
        super.onCleared()
    }

    private fun currentOriginal(): Bitmap? = when (val current = _state.value) {
        is AnalyzeState.Running -> current.original
        is AnalyzeState.Done -> current.original
        is AnalyzeState.Failed -> current.original
        else -> null
    }

    private fun cleanedFrame(): Bitmap? = (_state.value as? AnalyzeState.Done)?.cleaned

    private fun describe(error: Throwable): String =
        error.message?.takeIf { it.isNotBlank() } ?: error::class.java.simpleName

    private companion object {
        const val DEFAULT_INPUT_SIZE = 384
        const val SHA_PREFIX_LENGTH = 10
    }
}
