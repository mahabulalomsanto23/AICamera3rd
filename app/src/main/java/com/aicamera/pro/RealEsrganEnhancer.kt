package com.aicamera.pro

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Qualcomm Real-ESRGAN-General-x4v3, executed locally with TFLite CPU.
 * The network predicts 4x detail; we produce a practical 2x output by
 * downsampling its 4x tile, reducing memory versus a full 4x photograph.
 */
class RealEsrganEnhancer(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val outputShape: IntArray
    private val inputNhwc: Boolean
    private val outputNhwc: Boolean
    private val tile = 128
    private val overlap = 8
    private val stride = tile - overlap * 2

    init {
        val model = ModelAsset.map(context, "realesrgan_general_x4v3.tflite")
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(2) })
        inputShape = interpreter.getInputTensor(0).shape()
        outputShape = interpreter.getOutputTensor(0).shape()
        require(inputShape.size == 4 && inputShape[0] == 1) { "Unsupported Real-ESRGAN input: ${inputShape.contentToString()}" }
        require(outputShape.size == 4 && outputShape[0] == 1) { "Unsupported Real-ESRGAN output: ${outputShape.contentToString()}" }
        inputNhwc = inputShape[3] == 3
        outputNhwc = outputShape[3] == 3
        val ih = if (inputNhwc) inputShape[1] else inputShape[2]
        val iw = if (inputNhwc) inputShape[2] else inputShape[3]
        val oh = if (outputNhwc) outputShape[1] else outputShape[2]
        val ow = if (outputNhwc) outputShape[2] else outputShape[3]
        require(iw == tile && ih == tile && ow == tile * 4 && oh == tile * 4) {
            "Unexpected Real-ESRGAN scale: ${inputShape.contentToString()} -> ${outputShape.contentToString()}"
        }
        require(interpreter.getInputTensor(0).dataType() == DataType.FLOAT32 &&
            interpreter.getOutputTensor(0).dataType() == DataType.FLOAT32) {
            "Real-ESRGAN float model required"
        }
        Log.i(TAG, "MODEL_READY input=${inputShape.contentToString()} output=${outputShape.contentToString()} layout=${if(inputNhwc) "NHWC" else "NCHW"}")
    }

    /** Returns a 2x bitmap, or the original bitmap when the photo exceeds safe limits. */
    fun enhance(source: Bitmap, onProgress: ((Int, Int) -> Unit)? = null): Bitmap {
        val w = source.width
        val h = source.height
        if (w.toLong() * h.toLong() > MAX_INPUT_PIXELS) {
            Log.i(TAG, "SKIP image_too_large width=$w height=$h")
            return source
        }
        val outW = Math.multiplyExact(w, 2)
        val outH = Math.multiplyExact(h, 2)
        val result = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        val input = ByteBuffer.allocateDirect(tile * tile * 3 * 4).order(ByteOrder.nativeOrder())
        val output = ByteBuffer.allocateDirect(tile * 4 * tile * 4 * 3 * 4).order(ByteOrder.nativeOrder())
        val patch = IntArray(tile * tile)
        val sourceRow = IntArray(tile)
        val rgb = IntArray(tile * 4 * tile * 4)
        val canvas = android.graphics.Canvas(result)
        val drawPaint = android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG)
        val totalX = (w + stride - 1) / stride
        val totalY = (h + stride - 1) / stride
        val total = totalX * totalY
        var done = 0
        var inferenceMs = 0L
        val started = SystemClock.elapsedRealtime()
        try {
            var top = 0
            while (top < h) {
                var left = 0
                while (left < w) {
                    input.clear()
                    // Read one contiguous source row per tile row instead of
                    // 16,384 getPixel JNI calls per tile. Clamp-pad edge pixels.
                    val sourceX0 = (left - overlap).coerceAtLeast(0)
                    val sourceX1 = (left - overlap + tile).coerceAtMost(w)
                    val validWidth = (sourceX1 - sourceX0).coerceAtLeast(1)
                    for (y in 0 until tile) {
                        val sourceY = (top + y - overlap).coerceIn(0, h - 1)
                        source.getPixels(sourceRow, 0, validWidth, sourceX0, sourceY, validWidth, 1)
                        for (x in 0 until tile) {
                            val sourceX = (left + x - overlap).coerceIn(0, w - 1)
                            patch[y * tile + x] = sourceRow[(sourceX - sourceX0).coerceIn(0, validWidth - 1)]
                        }
                    }
                    if (inputNhwc) {
                        for (p in patch) {
                            input.putFloat(((p shr 16) and 255) / 255f)
                            input.putFloat(((p shr 8) and 255) / 255f)
                            input.putFloat((p and 255) / 255f)
                        }
                    } else {
                        for (shift in intArrayOf(16, 8, 0)) for (p in patch)
                            input.putFloat(((p shr shift) and 255) / 255f)
                    }
                    input.rewind()
                    output.clear()
                    val inferenceStarted = SystemClock.elapsedRealtime()
                    interpreter.run(input, output)
                    inferenceMs += SystemClock.elapsedRealtime() - inferenceStarted
                    output.rewind()
                    if (outputNhwc) {
                        for (i in rgb.indices) {
                            val r = (output.float.coerceIn(0f, 1f) * 255f).toInt()
                            val g = (output.float.coerceIn(0f, 1f) * 255f).toInt()
                            val b = (output.float.coerceIn(0f, 1f) * 255f).toInt()
                            rgb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        }
                    } else {
                        val planes = Array(3) { FloatArray(rgb.size) }
                        for (channel in 0..2) for (i in rgb.indices)
                            planes[channel][i] = output.float.coerceIn(0f, 1f)
                        for (i in rgb.indices) {
                            val r = (planes[0][i] * 255f).toInt()
                            val g = (planes[1][i] * 255f).toInt()
                            val b = (planes[2][i] * 255f).toInt()
                            rgb[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                        }
                    }
                    val tileBitmap = Bitmap.createBitmap(rgb, tile * 4, tile * 4, Bitmap.Config.ARGB_8888)
                    val crop = Bitmap.createBitmap(tileBitmap, overlap * 4, overlap * 4,
                        stride * 4, stride * 4)
                    val small = Bitmap.createScaledBitmap(crop, stride * 2, stride * 2, true)
                    val copyW = minOf(stride, w - left)
                    val copyH = minOf(stride, h - top)
                    val dst = Rect(left * 2, top * 2, (left + copyW) * 2, (top + copyH) * 2)
                    val srcRect = Rect(0, 0, copyW * 2, copyH * 2)
                    canvas.drawBitmap(small, srcRect, dst, drawPaint)
                    if (small !== crop && !small.isRecycled) small.recycle()
                    if (!crop.isRecycled) crop.recycle()
                    if (!tileBitmap.isRecycled) tileBitmap.recycle()
                    done++
                    onProgress?.invoke(done, total)
                    left += stride
                }
                top += stride
            }
            val detailStarted = SystemClock.elapsedRealtime()
            restoreSourceDetail(source, result)
            Log.i(TAG, "UPSCALE_DONE input=${w}x$h output=${outW}x$outH tiles=$total inference_ms=$inferenceMs detail_ms=${SystemClock.elapsedRealtime() - detailStarted} elapsed_ms=${SystemClock.elapsedRealtime() - started} thread_count=2")
            return result
        } catch (t: Throwable) {
            if (!result.isRecycled) result.recycle()
            throw t
        }
    }

    /**
     * Reintroduce restrained source high-frequency structure after SR. This helps
     * hair, fur, leaves and masonry retain edge character without strong sharpening.
     */
    private fun restoreSourceDetail(source: Bitmap, output: Bitmap) {
        val sw = source.width
        val sh = source.height
        val ow = output.width
        val rowCenter = IntArray(sw)
        val rowAbove = IntArray(sw)
        val rowBelow = IntArray(sw)
        val outputRow = IntArray(ow)
        for (y in 0 until output.height) {
            val sy = (y / 2).coerceIn(0, sh - 1)
            source.getPixels(rowCenter, 0, sw, 0, sy, sw, 1)
            source.getPixels(rowAbove, 0, sw, 0, (sy - 1).coerceAtLeast(0), sw, 1)
            source.getPixels(rowBelow, 0, sw, 0, (sy + 1).coerceAtMost(sh - 1), sw, 1)
            output.getPixels(outputRow, 0, ow, 0, y, ow, 1)
            for (x in 0 until ow) {
                val sx = (x / 2).coerceIn(0, sw - 1)
                val center = rowCenter[sx]
                val left = rowCenter[(sx - 1).coerceAtLeast(0)]
                val right = rowCenter[(sx + 1).coerceAtMost(sw - 1)]
                val up = rowAbove[sx]
                val down = rowBelow[sx]
                fun edge(shift: Int): Float {
                    val c = (center shr shift) and 255
                    val n = (((left shr shift) and 255) + ((right shr shift) and 255) +
                        ((up shr shift) and 255) + ((down shr shift) and 255)) * 0.25f
                    return (c - n).coerceIn(-12f, 12f)
                }
                val base = outputRow[x]
                val r = (((base shr 16) and 255) + edge(16) * 0.22f).toInt().coerceIn(0, 255)
                val g = (((base shr 8) and 255) + edge(8) * 0.22f).toInt().coerceIn(0, 255)
                val b = ((base and 255) + edge(0) * 0.22f).toInt().coerceIn(0, 255)
                outputRow[x] = (base and -0x1000000) or (r shl 16) or (g shl 8) or b
            }
            output.setPixels(outputRow, 0, ow, 0, y, ow, 1)
        }
    }
    override fun close() = interpreter.close()

    companion object {
        private const val TAG = "AICameraRealESRGAN"
        private const val MAX_INPUT_PIXELS = 1_500_000L
    }
}
