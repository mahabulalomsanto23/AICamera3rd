package com.aicamera.pro

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/**
 * ADE20K DeepLabV3-MobileNetV2 semantic masks.
 * Class IDs: sky=2, tree=4, grass=9, flower=66, palm=72, apparel=92.
 */
class SceneSegmenter(context: Context) : AutoCloseable {
    private val interpreter: Interpreter
    private val inputW: Int
    private val inputH: Int
    private val outputW: Int
    private val outputH: Int
    private val classes: Int

    init {
        val file = File(context.filesDir, "deeplabv3_ade20k.tflite")
        if (!file.exists() || file.length() == 0L) {
            context.assets.open("deeplabv3_ade20k.tflite").use { input ->
                FileOutputStream(file).use { input.copyTo(it) }
            }
        }
        interpreter = Interpreter(file, Interpreter.Options().apply { setNumThreads(4) })
        val ins = interpreter.getInputTensor(0).shape()
        val outs = interpreter.getOutputTensor(0).shape()
        require(ins.size == 4 && ins[0] == 1 && ins[3] == 3) { "Unsupported segmentation input: ${ins.contentToString()}" }
        require(outs[0] == 1 && (outs.size == 3 || (outs.size == 4 && (outs[3] == 1 || outs[3] >= 100)))) { "Unsupported ADE20K output: ${outs.contentToString()}" }
        inputH = ins[1]; inputW = ins[2]; outputH = outs[1]; outputW = outs[2]; classes = if (outs.size == 3 || outs[3] == 1) 150 else outs[3]
        require(classes > 92) { "ADE20K model has no apparel class" }
    }

    /** Applies restrained, class-specific treatment without globally shifting clothing colors. */
    fun apply(source: Bitmap, enhanced: Bitmap): Bitmap {
        if (source.width != enhanced.width || source.height != enhanced.height) return enhanced
        val small = Bitmap.createScaledBitmap(source, inputW, inputH, true)
        val rgb = IntArray(inputW * inputH)
        small.getPixels(rgb, 0, inputW, 0, 0, inputW, inputH)
        val tensor = interpreter.getInputTensor(0)
        val input = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        when (tensor.dataType()) {
            DataType.FLOAT32 -> for (p in rgb) {
                input.putFloat(((p shr 16 and 255) / 127.5f) - 1f)
                input.putFloat(((p shr 8 and 255) / 127.5f) - 1f)
                input.putFloat(((p and 255) / 127.5f) - 1f)
            }
            DataType.UINT8 -> for (p in rgb) {
                input.put(((p shr 16) and 255).toByte()); input.put(((p shr 8) and 255).toByte()); input.put((p and 255).toByte())
            }
            else -> return enhanced
        }
        input.rewind()
        val outTensor = interpreter.getOutputTensor(0)
        val output = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())
        interpreter.run(input, output)
        output.rewind()
        val labels = IntArray(outputW * outputH)
        if (outTensor.shape().size == 3 || outTensor.shape()[3] == 1) {
            for (p in labels.indices) {
                labels[p] = when (outTensor.dataType()) {
                    DataType.INT64 -> output.getLong().toInt()
                    DataType.INT32 -> output.getInt()
                    DataType.FLOAT32 -> output.getFloat().toInt()
                    DataType.UINT8 -> output.get().toInt() and 0xff
                    DataType.INT8 -> output.get().toInt()
                    else -> return enhanced
                }
            }
        } else {
            if (outTensor.dataType() != DataType.FLOAT32) return enhanced
            for (p in labels.indices) {
                var best = -Float.MAX_VALUE
                var bestClass = 0
                for (c in 0 until classes) {
                    val score = output.getFloat((p * classes + c) * 4)
                    if (score > best) { best = score; bestClass = c }
                }
                labels[p] = bestClass
            }
        }

        val w = enhanced.width; val h = enhanced.height
        val src = IntArray(w * h); val dst = IntArray(w * h)
        source.getPixels(src, 0, w, 0, 0, w, h); enhanced.getPixels(dst, 0, w, 0, 0, w, h)
        for (y in 0 until h) for (x in 0 until w) {
            val cls = labels[(y * outputH / h).coerceIn(0, outputH - 1) * outputW +
                (x * outputW / w).coerceIn(0, outputW - 1)]
            val i = y * w + x
            val original = src[i]; val changed = dst[i]
            val r = (changed shr 16) and 255; val g = (changed shr 8) and 255; val b = changed and 255
            val luma = (0.2126f * r + 0.7152f * g + 0.0722f * b)
            val satBoost = when (cls) {
                2 -> 1.10f                         // sky
                4, 9, 17, 72 -> 1.12f             // trees, grass, plants and palms
                66 -> 1.18f                       // flowers
                92 -> 0.82f                       // apparel: protect original color
                else -> 1.0f
            }
            val preserve = if (cls == 92) 0.22f else 0f
            val rr = (luma + (r - luma) * satBoost).coerceIn(0f,255f) * (1f-preserve) + ((original shr 16) and 255) * preserve
            val gg = (luma + (g - luma) * satBoost).coerceIn(0f,255f) * (1f-preserve) + ((original shr 8) and 255) * preserve
            val bb = (luma + (b - luma) * satBoost).coerceIn(0f,255f) * (1f-preserve) + (original and 255) * preserve
            dst[i] = (changed and -0x1000000) or (rr.toInt().coerceIn(0,255) shl 16) or
                (gg.toInt().coerceIn(0,255) shl 8) or bb.toInt().coerceIn(0,255)
        }
        return Bitmap.createBitmap(dst, w, h, Bitmap.Config.ARGB_8888)
    }


    /** ADE20K semantic class 12 is the person category. */
    fun personMask(source: Bitmap): Bitmap {
        val small = Bitmap.createScaledBitmap(source, inputW, inputH, true)
        val rgb = IntArray(inputW * inputH)
        small.getPixels(rgb, 0, inputW, 0, 0, inputW, inputH)
        val tensor = interpreter.getInputTensor(0)
        val input = ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        when (tensor.dataType()) {
            DataType.FLOAT32 -> for (p in rgb) {
                input.putFloat(((p shr 16 and 255) / 127.5f) - 1f)
                input.putFloat(((p shr 8 and 255) / 127.5f) - 1f)
                input.putFloat(((p and 255) / 127.5f) - 1f)
            }
            DataType.UINT8 -> for (p in rgb) {
                input.put(((p shr 16) and 255).toByte())
                input.put(((p shr 8) and 255).toByte())
                input.put((p and 255).toByte())
            }
            else -> throw IllegalStateException("Unsupported portrait model input")
        }
        input.rewind()
        val outTensor = interpreter.getOutputTensor(0)
        val output = ByteBuffer.allocateDirect(outTensor.numBytes()).order(ByteOrder.nativeOrder())
        interpreter.run(input, output)
        output.rewind()
        val labels = IntArray(outputW * outputH)
        if (outTensor.shape().size == 3 || outTensor.shape()[3] == 1) {
            for (p in labels.indices) {
                labels[p] = when (outTensor.dataType()) {
                    DataType.INT64 -> output.getLong().toInt()
                    DataType.INT32 -> output.getInt()
                    DataType.FLOAT32 -> output.getFloat().toInt()
                    DataType.UINT8 -> output.get().toInt() and 0xff
                    DataType.INT8 -> output.get().toInt()
                    else -> throw IllegalStateException("Unsupported portrait mask output")
                }
            }
        } else {
            if (outTensor.dataType() != DataType.FLOAT32) throw IllegalStateException("Unsupported portrait logits output")
            for (p in labels.indices) {
                var best = -Float.MAX_VALUE
                var bestClass = 0
                for (c in 0 until classes) {
                    val score = output.getFloat((p * classes + c) * 4)
                    if (score > best) { best = score; bestClass = c }
                }
                labels[p] = bestClass
            }
        }
        // ADE20K contiguous train-id 12 is person. Refuse unsafe masks rather than
        // turning the entire frame into a blurred background when detection fails.
        val personCount = labels.count { it == 12 }
        val personRatio = personCount.toFloat() / labels.size
        if (personRatio < 0.005f) {
            throw IllegalStateException("Portrait subject not detected. Try a clearer, full-body photo with the person visible.")
        }
        if (personRatio > 0.95f) {
            throw IllegalStateException("Portrait subject mask is unreliable for this photo. Please try another image.")
        }
        val maskPixels = IntArray(labels.size) { i -> if (labels[i] == 12) Color.WHITE else Color.BLACK }
        val lowMask = Bitmap.createBitmap(maskPixels, outputW, outputH, Bitmap.Config.ARGB_8888)
        return Bitmap.createScaledBitmap(lowMask, source.width, source.height, true)
    }

    override fun close() = interpreter.close()
}
