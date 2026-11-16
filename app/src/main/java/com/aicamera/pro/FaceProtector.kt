package com.aicamera.pro

import android.graphics.Bitmap
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import kotlin.math.max
import kotlin.math.min

/**
 * Bundled ML Kit face detector used as a preservation stage after MAXIM.
 * It preserves identity and natural skin color while transferring restrained
 * high-frequency facial texture from the source into the enhanced result.
 * Detection runs locally; no image is uploaded.
 */
class FaceProtector : AutoCloseable {
    private val detector: FaceDetector

    init {
        val options = FaceDetectorOptions.Builder()
            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
            .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
            .setMinFaceSize(0.06f)
            .build()
        detector = FaceDetection.getClient(options)
    }

    fun protect(source: Bitmap, enhanced: Bitmap): Bitmap {
        if (source.width != enhanced.width || source.height != enhanced.height) return enhanced
        val faces = try {
            Tasks.await(detector.process(InputImage.fromBitmap(source, 0)))
        } catch (_: Exception) {
            // Enhancement should remain usable if face detection fails.
            return enhanced
        }
        if (faces.isEmpty()) return enhanced

        val width = source.width
        val height = source.height
        val originalPixels = IntArray(width * height)
        val outputPixels = IntArray(width * height)
        source.getPixels(originalPixels, 0, width, 0, 0, width, height)
        enhanced.getPixels(outputPixels, 0, width, 0, 0, width, height)

        for (face in faces) {
            val bounds = face.boundingBox
            val cx = (bounds.left + bounds.right) * 0.5f
            val cy = (bounds.top + bounds.bottom) * 0.5f
            val rx = max(1f, bounds.width() * 0.62f)
            val ry = max(1f, bounds.height() * 0.72f)
            val left = max(0, (cx - rx).toInt())
            val right = min(width - 1, (cx + rx).toInt())
            val top = max(0, (cy - ry).toInt())
            val bottom = min(height - 1, (cy + ry).toInt())

            for (y in top..bottom) {
                val ny = (y - cy) / ry
                for (x in left..right) {
                    val nx = (x - cx) / rx
                    val distance = nx * nx + ny * ny
                    if (distance >= 1f) continue
                    // Soft elliptical mask: strongest preservation at face center,
                    // feathered toward the boundary to avoid visible rectangular edges.
                    val mask = ((1f - distance) / 0.35f).coerceIn(0f, 1f)
                    val originalWeight = 0.20f * mask
                    val index = y * width + x
                    val src = originalPixels[index]
                    val dst = outputPixels[index]
                    // Transfer only restrained high-frequency source detail. This
                    // restores pores, eyebrow/hair edges and facial definition
                    // without inventing facial features or changing identity.
                    val sx = x.coerceIn(1, width - 2)
                    val sy = y.coerceIn(1, height - 2)
                    fun channel(pixel: Int, shift: Int) = (pixel shr shift) and 255
                    fun detail(shift: Int): Float {
                        val center = channel(originalPixels[index], shift)
                        val neighbors = (
                            channel(originalPixels[sy * width + sx - 1], shift) +
                            channel(originalPixels[sy * width + sx + 1], shift) +
                            channel(originalPixels[(sy - 1) * width + sx], shift) +
                            channel(originalPixels[(sy + 1) * width + sx], shift)
                        ) * 0.25f
                        return ((center - neighbors) * 0.42f * mask).coerceIn(-10f, 10f)
                    }
                    val r = (((((dst shr 16) and 255) * (1f - originalWeight)) +
                        (((src shr 16) and 255) * originalWeight)) + detail(16)).toInt().coerceIn(0, 255)
                    val g = (((((dst shr 8) and 255) * (1f - originalWeight)) +
                        (((src shr 8) and 255) * originalWeight)) + detail(8)).toInt().coerceIn(0, 255)
                    val b = ((((dst and 255) * (1f - originalWeight)) +
                        ((src and 255) * originalWeight)) + detail(0)).toInt().coerceIn(0, 255)
                    outputPixels[index] = (dst and -0x1000000) or (r shl 16) or (g shl 8) or b
                }
            }
        }

        return Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also {
            it.setPixels(outputPixels, 0, width, 0, 0, width, height)
        }
    }

    override fun close() = detector.close()
}
