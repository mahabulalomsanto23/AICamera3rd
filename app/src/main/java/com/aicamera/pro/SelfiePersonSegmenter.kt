package com.aicamera.pro

import android.graphics.Bitmap
import android.graphics.Color
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.segmentation.Segmentation
import com.google.mlkit.vision.segmentation.selfie.SelfieSegmenterOptions
import com.google.android.gms.tasks.Tasks
import java.io.Closeable
import java.nio.ByteOrder
import kotlin.math.roundToInt

/** On-device ML Kit person segmentation for portrait background separation. */
class SelfiePersonSegmenter : Closeable {
    private val segmenter = Segmentation.getClient(
        SelfieSegmenterOptions.Builder()
            .setDetectorMode(SelfieSegmenterOptions.SINGLE_IMAGE_MODE)
            .build()
    )

    fun personMask(source: Bitmap): Bitmap {
        require(source.width >= 256 && source.height >= 256) {
            "Photo is too small for reliable portrait segmentation"
        }
        val input = InputImage.fromBitmap(source, 0)
        val mask = Tasks.await(segmenter.process(input))
        val width = mask.width
        val height = mask.height
        val buffer = mask.buffer.duplicate().order(ByteOrder.nativeOrder())
        buffer.rewind()
        val pixels = IntArray(width * height)
        var foreground = 0
        for (i in pixels.indices) {
            val confidence = buffer.float.coerceIn(0f, 1f)
            if (confidence >= 0.5f) foreground++
            val value = (confidence * 255f).roundToInt().coerceIn(0, 255)
            pixels[i] = Color.rgb(value, value, value)
        }
        val ratio = foreground.toFloat() / pixels.size
        require(ratio >= 0.005f && ratio <= 0.95f) {
            "Person not detected clearly. Try a brighter, sharper photo with the subject visible."
        }
        val low = Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
        val scaled = Bitmap.createScaledBitmap(low, source.width, source.height, true)
        if (scaled !== low) low.recycle()
        return scaled
    }

    override fun close() {
        segmenter.close()
    }
}
