package com.aicamera.pro

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import android.util.Log
import java.nio.FloatBuffer
import kotlin.math.min

/**
 * NTIRE 2024 RLFN baseline (MIT), executed fully offline through ONNX Runtime.
 *
 * RLFN is a dedicated learned 4x super-resolution network (0.317M parameters),
 * unlike the existing 256px MAXIM FiveK retouch model which mainly predicts tone.
 * 128px tiles are inferred and reduced to a practical 2x output for mobile memory.
 */
internal class RlfnEnhancer(context: Context) : AutoCloseable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val inputName: String
    private val tile = 128
    private val overlap = 8
    private val stride = tile - overlap * 2

    init {
        val model = ModelAsset.map(context, "rlfn_ntire2024_x4.onnx")
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(2)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = environment.createSession(model, options)
        inputName = session.inputNames.first()
        val inputInfo = session.inputInfo[inputName]?.info as? ai.onnxruntime.TensorInfo
            ?: error("RLFN input tensor metadata is unavailable")
        val outputInfo = session.outputInfo.values.first().info as? ai.onnxruntime.TensorInfo
            ?: error("RLFN output tensor metadata is unavailable")
        require(inputInfo.shape.contentEquals(longArrayOf(1, 3, 128, 128))) {
            "Unexpected RLFN input: ${inputInfo.shape.contentToString()}"
        }
        require(outputInfo.shape.contentEquals(longArrayOf(1, 3, 512, 512))) {
            "Unexpected RLFN output: ${outputInfo.shape.contentToString()}"
        }
        Log.i(TAG, "MODEL_READY input=1x3x128x128 output=1x3x512x512 runtime=ONNX_CPU")
    }

    fun enhance(source: Bitmap, onProgress: ((Int, Int) -> Unit)? = null): Bitmap {
        val width = source.width
        val height = source.height
        if (width.toLong() * height.toLong() > MAX_INPUT_PIXELS) return source

        val result = Bitmap.createBitmap(width * 2, height * 2, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(result)
        val paint = Paint(Paint.FILTER_BITMAP_FLAG)
        val patch = IntArray(tile * tile)
        val sourceRow = IntArray(tile)
        val input = FloatBuffer.allocate(3 * tile * tile)
        val totalX = (width + stride - 1) / stride
        val totalY = (height + stride - 1) / stride
        val total = totalX * totalY
        var completed = 0
        val started = android.os.SystemClock.elapsedRealtime()
        try {
            var top = 0
            while (top < height) {
                var left = 0
                while (left < width) {
                    val sourceX0 = (left - overlap).coerceAtLeast(0)
                    val sourceX1 = (left - overlap + tile).coerceAtMost(width)
                    val validWidth = (sourceX1 - sourceX0).coerceAtLeast(1)
                    for (y in 0 until tile) {
                        val sy = (top + y - overlap).coerceIn(0, height - 1)
                        source.getPixels(sourceRow, 0, validWidth, sourceX0, sy, validWidth, 1)
                        for (x in 0 until tile) {
                            val sx = (left + x - overlap).coerceIn(0, width - 1)
                            patch[y * tile + x] = sourceRow[(sx - sourceX0).coerceIn(0, validWidth - 1)]
                        }
                    }

                    input.clear()
                    for (shift in intArrayOf(16, 8, 0)) {
                        for (pixel in patch) input.put(((pixel shr shift) and 255).toFloat())
                    }
                    input.rewind()

                    val tileBitmap = OnnxTensor.createTensor(
                        environment, input, longArrayOf(1, 3, tile.toLong(), tile.toLong())
                    )
                    val outputs = try {
                        session.run(mapOf(inputName to tileBitmap))
                    } finally {
                        tileBitmap.close()
                    }
                    val values = try {
                        val tensor = outputs[0] as OnnxTensor
                        tensor.value as Array<Array<Array<FloatArray>>>
                    } finally {
                        outputs.close()
                    }
                    val pixels = IntArray(tile * 4 * tile * 4)
                    val planeSize = tile * 4 * tile * 4
                    for (i in pixels.indices) {
                        val r = values[0][0][i / (tile * 4)][i % (tile * 4)].toInt().coerceIn(0, 255)
                        val g = values[0][1][i / (tile * 4)][i % (tile * 4)].toInt().coerceIn(0, 255)
                        val b = values[0][2][i / (tile * 4)][i % (tile * 4)].toInt().coerceIn(0, 255)
                        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                    }
                    val tileOutput = Bitmap.createBitmap(pixels, tile * 4, tile * 4, Bitmap.Config.ARGB_8888)
                    val crop = Bitmap.createBitmap(tileOutput, overlap * 4, overlap * 4,
                        stride * 4, stride * 4)
                    val half = Bitmap.createScaledBitmap(crop, stride * 2, stride * 2, true)
                    val copyW = min(stride, width - left)
                    val copyH = min(stride, height - top)
                    canvas.drawBitmap(
                        half,
                        Rect(0, 0, copyW * 2, copyH * 2),
                        Rect(left * 2, top * 2, (left + copyW) * 2, (top + copyH) * 2),
                        paint
                    )
                    if (half !== crop && !half.isRecycled) half.recycle()
                    if (!crop.isRecycled) crop.recycle()
                    if (!tileOutput.isRecycled) tileOutput.recycle()

                    completed++
                    onProgress?.invoke(completed, total)
                    left += stride
                }
                top += stride
            }
            Log.i(TAG, "RLFN_DONE input=${width}x${height} output=${width * 2}x${height * 2} tiles=$total elapsed_ms=${android.os.SystemClock.elapsedRealtime() - started}")
            return result
        } catch (t: Throwable) {
            if (!result.isRecycled) result.recycle()
            throw t
        }
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "AICameraRLFN"
        private const val MAX_INPUT_PIXELS = 1_500_000L
    }
}
