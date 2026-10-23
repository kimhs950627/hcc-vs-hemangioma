package com.kimhs.liverdx.util

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.File
import java.io.InputStream

/** Decodes a picked or captured image, upright and small enough to clean up cheaply. */
object ImageLoading {

    /**
     * Decoded at twice the model input so the resampler still has real pixels to work with; decoding
     * straight to the model size would alias before the cleanup even runs.
     */
    private const val DECODE_FACTOR = 2

    fun loadFromUri(context: Context, uri: Uri, modelInputSize: Int): Bitmap =
        decode(
            openStream = { context.contentResolver.openInputStream(uri) },
            modelInputSize = modelInputSize,
            description = uri.toString(),
        )

    fun loadFromFile(file: File, modelInputSize: Int): Bitmap =
        decode(
            openStream = { file.inputStream() },
            modelInputSize = modelInputSize,
            description = file.name,
        )

    private fun decode(
        openStream: () -> InputStream?,
        modelInputSize: Int,
        description: String,
    ): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        openStream()?.use { BitmapFactory.decodeStream(it, null, bounds) }
        require(bounds.outWidth > 0 && bounds.outHeight > 0) { "이미지를 읽을 수 없습니다: $description" }

        val options = BitmapFactory.Options().apply {
            inSampleSize = sampleSize(bounds.outWidth, bounds.outHeight, modelInputSize * DECODE_FACTOR)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        val decoded = openStream()?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: error("이미지 디코딩에 실패했습니다: $description")

        return applyExifRotation(openStream, decoded)
    }

    /** Largest power-of-two subsample that still keeps both sides at or above [requested]. */
    private fun sampleSize(width: Int, height: Int, requested: Int): Int {
        var sample = 1
        while (width / (sample * 2) >= requested && height / (sample * 2) >= requested) {
            sample *= 2
        }
        return sample
    }

    /**
     * Applies the EXIF orientation through the library's own decomposition, which also covers the
     * mirrored-transpose cases (5 and 7) that phone cameras emit.
     */
    private fun applyExifRotation(openStream: () -> InputStream?, bitmap: Bitmap): Bitmap {
        val exif = openStream()?.use { ExifInterface(it) } ?: return bitmap
        val rotation = exif.rotationDegrees
        val flipped = exif.isFlipped
        if (rotation == 0 && !flipped) return bitmap

        val matrix = Matrix()
        if (flipped) matrix.postScale(-1f, 1f)
        if (rotation != 0) matrix.postRotate(rotation.toFloat())

        val upright = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (upright !== bitmap) bitmap.recycle()
        return upright
    }
}

/**
 * One private cache file that the in-app camera overwrites on every shot, deleted as soon as the
 * frame has been decoded, so no ultrasound image outlives the analysis that used it.
 */
object CaptureFiles {
    private const val DIRECTORY = "captures"
    private const val FILE_NAME = "capture.jpg"

    fun nextFile(context: Context): File =
        File(File(context.cacheDir, DIRECTORY).apply { mkdirs() }, FILE_NAME)

    /** Deletes the capture file; safe to call whether or not one exists. */
    fun delete(context: Context) {
        nextFile(context).delete()
    }
}
