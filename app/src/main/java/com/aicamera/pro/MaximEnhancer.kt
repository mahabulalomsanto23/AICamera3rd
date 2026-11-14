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
        // MAXIM exposes intermediate supervision outputs as well as its final prediction.
        // Match the reference inference pipeline: use the LAST full-resolution RGB output.
        val fullOutputIndices = (0 until interpreter.outputTensorCount).filter { index ->
            interpreter.getOutputTensor(index).shape().contentEquals(intArrayOf(1, 256, 256, 3))
        }
        val fullOutputIndex = fullOutputIndices.lastOrNull()
        require(fullOutputIndex != null) {
            "MAXIM has no full-resolution output. Outputs: " +
                (0 until interpreter.outputTensorCount).joinToString { interpreter.getOutputTensor(it).shape().contentToString() }
        }
        outputIndex = fullOutputIndex
        Log.i(TAG, "MODEL_TENSORS input_type=${interpreter.getInputTensor(0).dataType()} outputs=" +
            (0 until interpreter.outputTensorCount).joinToString { index ->
                val tensor = interpreter.getOutputTensor(index)
                "$index:${tensor.name()}:${tensor.shape().contentToString()}:${tensor.dataType()}"
            } + " selected_output=$outputIndex")
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
            // The published MAXIM FiveK inference pipeline converts RGB pixels to
            // float [0, 1] before invoking the model. Feeding 0..255 floats here
            // pushes the network far outside its training range and causes severe
            // colour corruption.
            DataType.FLOAT32 -> sourcePixels.forEach { c ->
                input.putFloat(((c shr 16) and 255) / 255f)
                input.putFloat(((c shr 8) and 255) / 255f)
                input.putFloat((c and 255) / 255f)
            }
            DataType.UINT8 -> {
                val q = inputTensor.quantizationParams()
                require(q.scale > 0f) { "MAXIM UINT8 input has invalid quantization scale: ${q.scale}" }
                sourcePixels.forEach { c ->
                    fun quantize(channel: Int): Byte =
                        ((channel / 255f / q.scale) + q.zeroPoint).toInt().coerceIn(0, 255).toByte()
                    input.put(quantize((c shr 16) and 255))
                    input.put(quantize((c shr 8) and 255))
                    input.put(quantize(c and 255))
                }
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
            // MAXIM FiveK returns RGB values in [0, 1]. Convert back to
            // 8-bit sRGB only after clipping, as in the reference implementation.
            DataType.UINT8 -> {
                val q = outputTensor.quantizationParams()
                require(q.scale > 0f) { "MAXIM UINT8 output has invalid quantization scale: ${q.scale}" }
                for (i in enhancedSmall.indices) {
                    fun dequantize(): Int {
                        val raw = output.get().toInt() and 255
                        val normalized = (raw - q.zeroPoint) * q.scale
                        return (normalized.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                    }
                    val r = dequantize()
                    val g = dequantize()
                    val b = dequantize()
                    enhancedSmall[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            DataType.FLOAT32 -> {
                for (i in enhancedSmall.indices) {
                    val r = (output.float.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                    val g = (output.float.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
                    val b = (output.float.coerceIn(0f, 1f) * 255f).toInt().coerceIn(0, 255)
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
