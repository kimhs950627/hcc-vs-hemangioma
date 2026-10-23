package com.kimhs.liverdx.ml

import android.content.Context
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.security.MessageDigest

/** Validation-set operating points (Youden's J) shipped with the model. */
data class Cutoffs(val confidence: Float, val hccCosine: Float, val delta: Float)

/** Asset file names of the prototype banks. */
data class PrototypeFiles(val hccMean: String, val hemMean: String, val hccKmeans: String)

/** Parsed `assets/model_meta.json` — the single source of truth for constants. */
data class ModelMeta(
    val modelAsset: String,
    val precision: String,
    val modelBytes: Long,
    val cosineDrift: Float,
    val inputWidth: Int,
    val inputHeight: Int,
    val labels: List<String>,
    val positiveIndex: Int,
    val patchGrid: Int,
    val embedDim: Int,
    val prototypes: PrototypeFiles,
    val cutoffs: Cutoffs,
    val training: String,
    val preprocessOrder: List<String>,
) {
    companion object {
        fun fromJson(json: JSONObject): ModelMeta {
            val labelArray = json.getJSONArray("labels")
            val labels = List(labelArray.length()) { labelArray.getString(it) }
            val positiveLabel = json.getString("positive_label")
            val input = json.getJSONObject("input")
            val prototypes = json.getJSONObject("prototypes")
            val cutoffs = json.getJSONObject("cutoffs")
            return ModelMeta(
                modelAsset = json.getString("model_asset"),
                precision = json.getString("precision"),
                modelBytes = json.getLong("model_bytes"),
                cosineDrift = json.getJSONObject("precision_drift").getDouble("cosine").toFloat(),
                inputWidth = input.getInt("width"),
                inputHeight = input.getInt("height"),
                labels = labels,
                positiveIndex = labels.indexOf(positiveLabel).coerceAtLeast(0),
                patchGrid = json.getInt("patch_grid"),
                embedDim = json.getInt("embed_dim"),
                prototypes = PrototypeFiles(
                    hccMean = prototypes.getString("hcc_mean"),
                    hemMean = prototypes.getString("hem_mean"),
                    hccKmeans = prototypes.getString("hcc_kmeans"),
                ),
                cutoffs = Cutoffs(
                    confidence = cutoffs.getDouble("confidence_score").toFloat(),
                    hccCosine = cutoffs.getDouble("hcc_cosine_score").toFloat(),
                    delta = cutoffs.getDouble("delta_score").toFloat(),
                ),
                training = json.getString("training"),
                preprocessOrder = json.getJSONArray("preprocess_order").let { order ->
                    List(order.length()) { order.getString(it) }
                },
            )
        }
    }
}

/**
 * Everything read from `assets/` once per process: the memory-mapped flatbuffer, the metadata and
 * the prototype banks.
 */
class ModelBundle(
    val modelBuffer: ByteBuffer,
    val meta: ModelMeta,
    val modelSha256: String,
    val hccMean: FloatArray,
    val hemMean: FloatArray,
    val hccKmeans: Array<FloatArray>,
) {
    companion object {
        private const val META_ASSET = "model_meta.json"
        private const val BYTES_PER_FLOAT = 4

        fun load(context: Context): ModelBundle {
            val metaJson = context.assets.open(META_ASSET).use { it.readBytes().decodeToString() }
            val meta = ModelMeta.fromJson(JSONObject(metaJson))
            return ModelBundle(
                modelBuffer = mapAsset(context, meta.modelAsset),
                meta = meta,
                modelSha256 = sha256(context, meta.modelAsset),
                hccMean = readFloats(context, meta.prototypes.hccMean),
                hemMean = readFloats(context, meta.prototypes.hemMean),
                hccKmeans = readMatrix(context, meta.prototypes.hccKmeans, meta.embedDim),
            )
        }

        /** Maps the uncompressed asset straight into memory; TFLite reads it without a copy. */
        private fun mapAsset(context: Context, name: String): ByteBuffer =
            context.assets.openFd(name).use { descriptor ->
                FileInputStream(descriptor.fileDescriptor).channel
                    .map(FileChannel.MapMode.READ_ONLY, descriptor.startOffset, descriptor.declaredLength)
                    .order(ByteOrder.nativeOrder())
            }

        private fun sha256(context: Context, name: String): String {
            val digest = MessageDigest.getInstance("SHA-256")
            context.assets.open(name).use { stream ->
                val chunk = ByteArray(1 shl 16)
                while (true) {
                    val read = stream.read(chunk)
                    if (read <= 0) break
                    digest.update(chunk, 0, read)
                }
            }
            return digest.digest().joinToString("") { "%02x".format(it.toInt() and 0xFF) }
        }

        private fun readFloats(context: Context, name: String): FloatArray {
            val bytes = context.assets.open(name).use { it.readBytes() }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(bytes.size / BYTES_PER_FLOAT) { buffer.getFloat() }
        }

        private fun readMatrix(context: Context, name: String, dim: Int): Array<FloatArray> {
            val flat = readFloats(context, name)
            return Array(flat.size / dim) { row -> FloatArray(dim) { col -> flat[row * dim + col] } }
        }
    }
}
