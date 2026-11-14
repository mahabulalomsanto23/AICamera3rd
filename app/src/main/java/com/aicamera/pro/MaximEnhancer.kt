package com.aicamera.pro

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Runs the pretrained Google MAXIM-S2 FiveK retouching network locally with LiteRT/TFLite.
 * The network is evaluated at its exported 256x256 input size. Its low-resolution
 * color/tone correction is smoothly transferred to the original-resolution photo,
 * preserving the original high-frequency texture and image dimensions.
 */
class MaximEnhancer(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private var outputIndex: Int = 0

    init {
        val modelBuffer = ModelAsset.map(context, "maxim_fivek_fp16.tflite")
        // Limit CPU contention and temporary tensor pressure on Snapdragon 865.
        val options = Interpreter.Options().apply { setNumThreads(2) }
        interpreter = Interpreter(modelBuffer, options)
        val inputShape = interpreter.getInputTensor(0).shape()
        require(inputShape.contentEquals(intArrayOf(1, 256, 256, 3))) {
            "Unexpected MAXIM input shape: ${inputShape.contentToString()}"
        }
        val fullOutputIndex = (0 until interpreter.outputTensorCount).firstOrNull { index ->
            interpreter.getOutputTensor(index).shape().contentEquals(intArrayOf(1, 256, 256, 3))
        }
        require(fullOutputIndex != null) {
            "MAXIM has no full-resolution output. Outputs: " +
                (0 until interpreter.outputTensorCount).joinToString { interpreter.getOutputTensor(it).shape().contentToString() }
        }
        outputIndex = fullOutputIndex
    }

    fun enhance(source: Bitmap, onProgress: ((Int, Int) -> Unit)? = null): Bitmap {
        val original = if (source.config == Bitmap.Config.ARGB_8888) source else source.copy(Bitmap.Config.ARGB_8888, false)
        val small = Bitmap.createScaledBitmap(original, 256, 256, true)
        val sourcePixels = IntArray(256 * 256)
        small.getPixels(sourcePixels, 0, 256, 0, 0, 256, 256)
        if (small !== original && !small.isRecycled) small.recycle()

        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(outputIndex)
        val input = ByteBuffer.allocateDirect(inputTensor.numBytes()).order(ByteOrder.nativeOrder())
        when (inputTensor.dataType()) {
            DataType.UINT8 -> sourcePixels.forEach { c ->
                input.put(((c shr 16) and 255).toByte())
                input.put(((c shr 8) and 255).toByte())
                input.put((c and 255).toByte())
            }
            DataType.FLOAT32 -> sourcePixels.forEach { c ->
                input.putFloat(((c shr 16) and 255).toFloat())
                input.putFloat(((c shr 8) and 255).toFloat())
                input.putFloat((c and 255).toFloat())
            }
            else -> error("Unsupported MAXIM input type: ${inputTensor.dataType()}")
        }
        input.rewind()

        val output = ByteBuffer.allocateDirect(outputTensor.numBytes()).order(ByteOrder.nativeOrder())
        val inferenceStarted = SystemClock.elapsedRealtime()
        Log.i(TAG, "TFLITE_INFERENCE_START input_bytes=${inputTensor.numBytes()} output_bytes=${outputTensor.numBytes()} threads=2")
        interpreter.runForMultipleInputsOutputs(arrayOf(input), mapOf(outputIndex to output))
        Log.i(TAG, "TFLITE_INFERENCE_DONE elapsed_ms=${SystemClock.elapsedRealtime() - inferenceStarted}")
        output.rewind()

        val enhancedSmall = IntArray(256 * 256)
        when (outputTensor.dataType()) {
            DataType.UINT8 -> {
                val q = outputTensor.quantizationParams()
                for (i in enhancedSmall.indices) {
                    val r = (((output.get().toInt() and 255) - q.zeroPoint) * q.scale).toInt().coerceIn(0, 255)
                    val g = (((output.get().toInt() and 255) - q.zeroPoint) * q.scale).toInt().coerceIn(0, 255)
                    val b = (((output.get().toInt() and 255) - q.zeroPoint) * q.scale).toInt().coerceIn(0, 255)
                    enhancedSmall[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            DataType.FLOAT32 -> {
                val values = FloatArray(256 * 256 * 3)
                var maxValue = 0f
                for (i in values.indices) {
                    values[i] = output.float
                    if (values[i] > maxValue) maxValue = values[i]
                }
                val scale = if (maxValue <= 1.5f) 255f else 1f
                for (i in enhancedSmall.indices) {
                    val r = (values[i * 3] * scale).toInt().coerceIn(0, 255)
                    val g = (values[i * 3 + 1] * scale).toInt().coerceIn(0, 255)
                    val b = (values[i * 3 + 2] * scale).toInt().coerceIn(0, 255)
                    enhancedSmall[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            else -> error("Unsupported MAXIM output type: ${outputTensor.dataType()}")
        }

        // Transfer the learned color/tone residual without allocating multiple
        // full-resolution IntArrays. The old implementation held two large pixel
        // arrays plus two full-size bitmaps at once, which could OOM on 8GB phones.
        val deltaPixels = IntArray(enhancedSmall.size)
        for (i in deltaPixels.indices) {
            val a = sourcePixels[i]
            val b = enhancedSmall[i]
            val dr = (((b shr 16) and 255) - ((a shr 16) and 255) + 128).coerceIn(0, 255)
            val dg = (((b shr 8) and 255) - ((a shr 8) and 255) + 128).coerceIn(0, 255)
            val db = ((b and 255) - (a and 255) + 128).coerceIn(0, 255)
            deltaPixels[i] = (0xFF shl 24) or (dr shl 16) or (dg shl 8) or db
        }
        // Bilinearly sample the 256px residual per output row. This avoids
        // allocating a second full-resolution bitmap for the upscaled residual.
        val result = Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888)
        val baseRow = IntArray(original.width)
        val outRow = IntArray(original.width)
        try {
            for (y in 0 until original.height) {
                original.getPixels(baseRow, 0, original.width, 0, y, original.width, 1)
                val fy = (y * 255f / maxOf(1, original.height - 1)).coerceIn(0f, 255f)
                val y0 = fy.toInt()
                val y1 = (y0 + 1).coerceAtMost(255)
                val wy = fy - y0
                for (x in 0 until original.width) {
                    val fx = (x * 255f / maxOf(1, original.width - 1)).coerceIn(0f, 255f)
                    val x0 = fx.toInt()
                    val x1 = (x0 + 1).coerceAtMost(255)
                    val wx = fx - x0
                    val i00 = deltaPixels[y0 * 256 + x0]
                    val i10 = deltaPixels[y0 * 256 + x1]
                    val i01 = deltaPixels[y1 * 256 + x0]
                    val i11 = deltaPixels[y1 * 256 + x1]
                    val c = baseRow[x]
                    fun residual(shift: Int): Int {
                        val a = ((i00 shr shift) and 255) * (1f - wx) + ((i10 shr shift) and 255) * wx
                        val b = ((i01 shr shift) and 255) * (1f - wx) + ((i11 shr shift) and 255) * wx
                        return (a * (1f - wy) + b * wy).toInt()
                    }
                    val r = (((c shr 16) and 255) + residual(16) - 128).coerceIn(0, 255)
                    val g = (((c shr 8) and 255) + residual(8) - 128).coerceIn(0, 255)
                    val b = ((c and 255) + residual(0) - 128).coerceIn(0, 255)
                    outRow[x] = (c and -0x1000000) or (r shl 16) or (g shl 8) or b
                }
                result.setPixels(outRow, 0, original.width, 0, y, original.width, 1)
                if (onProgress != null && (y == original.height - 1 || y % 24 == 0)) onProgress(y + 1, original.height)
            }
        } catch (t: Throwable) {
            if (!result.isRecycled) result.recycle()
            throw t
        } finally {
            if (original !== source && !original.isRecycled) original.recycle()
        }
        return result
    }

    override fun close() = interpreter.close()

    private companion object {
        const val TAG = "AICameraMaxim"
    }
}
