package com.kimhs.liverdx.ml

import android.os.SystemClock
import org.tensorflow.lite.Interpreter
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** One inference result, with every marker the paper defines. */
data class Analysis(
    val label: String,
    val isMalignant: Boolean,
    val confidence: Float,
    val benignProbability: Float,
    val hccCosine: Float,
    val hemCosine: Float,
    val delta: Float,
    val cutoffs: Cutoffs,
    val byConfidence: Boolean,
    val byHccCosine: Boolean,
    val byDelta: Boolean,
    val attentionMap: FloatArray,
    val prototypeMap: FloatArray,
    val grid: Int,
    val inferenceMillis: Long,
) {
    /** All three markers land on the same side of their cutoffs. */
    val unanimous: Boolean get() = byConfidence == byHccCosine && byHccCosine == byDelta
}

/**
 * TFLite wrapper around the exported flatbuffer.
 *
 * Output tensors are matched by shape, not by name: the converter names them `Identity`,
 * `Identity_1`, ... and the order is not contractual.
 */
class Classifier(private val bundle: ModelBundle, threadCount: Int = 4) : Closeable {

    private val meta = bundle.meta
    private val interpreter = Interpreter(bundle.modelBuffer, Interpreter.Options().setNumThreads(threadCount))

    private val attentionKey = "attention"
    private val patchesKey = "patches"
    private val embeddingKey = "embedding"
    private val probsKey = "probs"

    /** Semantic name per expected output shape. */
    private val outputShapes by lazy {
        mapOf(
            listOf(1, meta.patchGrid, meta.patchGrid, 1) to attentionKey,
            listOf(1, meta.patchGrid * meta.patchGrid, meta.embedDim) to patchesKey,
            listOf(1, meta.embedDim) to embeddingKey,
            listOf(1, meta.labels.size) to probsKey,
        )
    }

    fun analyze(modelInput: FloatArray): Analysis {
        require(meta.labels.size == BINARY_CLASSES) {
            "this decoder assumes a binary head, model_meta.json declares ${meta.labels.size} labels"
        }
        require(modelInput.size == meta.inputWidth * meta.inputHeight * RGB_CHANNELS) {
            "model input holds ${modelInput.size} values, expected ${meta.inputWidth * meta.inputHeight * RGB_CHANNELS}"
        }
        val input = ByteBuffer.allocateDirect(modelInput.size * BYTES_PER_FLOAT)
            .order(ByteOrder.nativeOrder())
        input.asFloatBuffer().put(modelInput)
        input.rewind()

        val tensors = HashMap<String, Any>(outputShapes.size)
        val outputs = HashMap<Int, Any>(outputShapes.size)
        for (index in 0 until interpreter.outputTensorCount) {
            val shape = interpreter.getOutputTensor(index).shape()
            val key = outputShapes[shape.toList()]
                ?: error("unexpected flatbuffer output shape ${shape.toList()}")
            val allocation = allocateOutput(shape)
            tensors[key] = allocation
            outputs[index] = allocation
        }

        val started = SystemClock.elapsedRealtime()
        interpreter.runForMultipleInputsOutputs(arrayOf(input), outputs)
        val elapsed = SystemClock.elapsedRealtime() - started

        val probs = floats2D(tensors[probsKey], probsKey)
        val embedding = floats2D(tensors[embeddingKey], embeddingKey)[0]
        val patches = floats3D(tensors[patchesKey], patchesKey)[0]
        val attention = floats4D(tensors[attentionKey], attentionKey)[0]

        val grid = meta.patchGrid
        val hccCosine = ScoreMath.cosine(embedding, bundle.hccMean)
        val hemCosine = ScoreMath.cosine(embedding, bundle.hemMean)
        val delta = hccCosine - hemCosine
        val confidence = probs[0][meta.positiveIndex]

        val flatAttention = FloatArray(grid * grid) { attention[it / grid][it % grid][0] }
        val flatPatches = FloatArray(grid * grid * meta.embedDim)
        for (patch in 0 until grid * grid) {
            System.arraycopy(patches[patch], 0, flatPatches, patch * meta.embedDim, meta.embedDim)
        }

        return Analysis(
            label = meta.labels[meta.positiveIndex],
            isMalignant = confidence >= meta.cutoffs.confidence,
            confidence = confidence,
            benignProbability = probs[0][1 - meta.positiveIndex],
            hccCosine = hccCosine,
            hemCosine = hemCosine,
            delta = delta,
            cutoffs = meta.cutoffs,
            byConfidence = confidence >= meta.cutoffs.confidence,
            byHccCosine = hccCosine >= meta.cutoffs.hccCosine,
            byDelta = delta >= meta.cutoffs.delta,
            attentionMap = ScoreMath.minMaxNormalize(flatAttention),
            prototypeMap = ScoreMath.prototypeMap(flatPatches, bundle.hccKmeans, grid),
            grid = grid,
            inferenceMillis = elapsed,
        )
    }

    /** Nested float array of exactly [shape], which is what TFLite's Java API expects. */
    private fun allocateOutput(shape: IntArray): Any = allocateRank(shape, 0)

    private fun allocateRank(shape: IntArray, axis: Int): Any =
        if (axis == shape.size - 1) {
            FloatArray(shape[axis])
        } else {
            Array(shape[axis]) { allocateRank(shape, axis + 1) }
        }

    @Suppress("UNCHECKED_CAST")
    private fun floats2D(value: Any?, key: String): Array<FloatArray> =
        value as? Array<FloatArray> ?: error("missing 2-D output '$key'")

    @Suppress("UNCHECKED_CAST")
    private fun floats3D(value: Any?, key: String): Array<Array<FloatArray>> =
        value as? Array<Array<FloatArray>> ?: error("missing 3-D output '$key'")

    @Suppress("UNCHECKED_CAST")
    private fun floats4D(value: Any?, key: String): Array<Array<Array<FloatArray>>> =
        value as? Array<Array<Array<FloatArray>>> ?: error("missing 4-D output '$key'")

    override fun close() = interpreter.close()

    private companion object {
        const val BYTES_PER_FLOAT = 4
        const val RGB_CHANNELS = 3
        const val BINARY_CLASSES = 2
    }
}
