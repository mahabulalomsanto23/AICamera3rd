package com.aicamera.pro

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
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
        val modelFile = File(context.filesDir, "maxim_fivek_fp16.tflite")
        if (!modelFile.exists() || modelFile.length() == 0L) {
            context.assets.open("maxim_fivek_fp16.tflite").use { input ->
                FileOutputStream(modelFile).use { output -> input.copyTo(output) }
            }
        }
        val options = Interpreter.Options().apply { setNumThreads(4) }
        interpreter = Interpreter(modelFile, options)
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

    fun enhance(source: Bitmap): Bitmap {
        val original = if (source.config == Bitmap.Config.ARGB_8888) source else source.copy(Bitmap.Config.ARGB_8888, false)
        val small = Bitmap.createScaledBitmap(original, 256, 256, true)
        val sourcePixels = IntArray(256 * 256)
        small.getPixels(sourcePixels, 0, 256, 0, 0, 256, 256)

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
        interpreter.runForMultipleInputsOutputs(arrayOf(input), mapOf(outputIndex to output))
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

        // Transfer the model's learned color/tone residual to the full-resolution image.
        // This keeps original fine detail instead of enlarging a 256px result.
        val deltaPixels = IntArray(enhancedSmall.size)
        for (i in deltaPixels.indices) {
            val a = sourcePixels[i]
            val b = enhancedSmall[i]
            val dr = (((b shr 16) and 255) - ((a shr 16) and 255) + 128).coerceIn(0, 255)
            val dg = (((b shr 8) and 255) - ((a shr 8) and 255) + 128).coerceIn(0, 255)
            val db = ((b and 255) - (a and 255) + 128).coerceIn(0, 255)
            deltaPixels[i] = (0xFF shl 24) or (dr shl 16) or (dg shl 8) or db
        }
        val deltaBitmap = Bitmap.createBitmap(deltaPixels, 256, 256, Bitmap.Config.ARGB_8888)
        val fullDelta = Bitmap.createScaledBitmap(deltaBitmap, original.width, original.height, true)
        val deltas = IntArray(original.width * original.height)
        val base = IntArray(deltas.size)
        fullDelta.getPixels(deltas, 0, original.width, 0, 0, original.width, original.height)
        original.getPixels(base, 0, original.width, 0, 0, original.width, original.height)
        for (i in base.indices) {
            val c = base[i]
            val d = deltas[i]
            val r = (((c shr 16) and 255) + ((d shr 16) and 255) - 128).coerceIn(0, 255)
            val g = (((c shr 8) and 255) + ((d shr 8) and 255) - 128).coerceIn(0, 255)
            val b = ((c and 255) + (d and 255) - 128).coerceIn(0, 255)
            base[i] = (c and -0x1000000) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(original.width, original.height, Bitmap.Config.ARGB_8888).also {
            it.setPixels(base, 0, original.width, 0, 0, original.width, original.height)
        }
    }

    override fun close() = interpreter.close()
}
