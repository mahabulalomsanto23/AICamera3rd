package com.aicamera.pro

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
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
        val modelBuffer = ModelAsset.map(context, "deeplabv3_ade20k.tflite")
        interpreter = Interpreter(modelBuffer, Interpreter.Options().apply { setNumThreads(2) })
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
        if (small !== source && !small.isRecycled) small.recycle()
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

        // Precompute source luminance once. Besides being faster on Snapdragon 865,
        // this gives the detail pass a stable reference that is not affected by MAXIM.
        val sourceY = FloatArray(w * h)
        for (i in src.indices) {
            val p = src[i]
            sourceY[i] = 0.2126f * ((p shr 16) and 255) +
                0.7152f * ((p shr 8) and 255) + 0.0722f * (p and 255)
        }

        for (y in 0 until h) for (x in 0 until w) {
            val cls = labels[(y * outputH / h).coerceIn(0, outputH - 1) * outputW +
                (x * outputW / w).coerceIn(0, outputW - 1)]
            val i = y * w + x
            val original = src[i]; val changed = dst[i]
            val r = (changed shr 16) and 255; val g = (changed shr 8) and 255; val b = changed and 255
            val luma = 0.2126f * r + 0.7152f * g + 0.0722f * b

            // Semantic photographic profiles: exposure/contrast and chroma are
            // selected by the detected object, not applied as one global filter.
            // Values are deliberately restrained to avoid neon greens or cyan skies.
            val saturation: Float
            val contrast: Float
            val exposure: Float
            val detailGain: Float
            val preserveOriginal: Float
            when (cls) {
                2 -> { // sky: protect highlights, gently restore blue/cyan
                    saturation = 1.075f; contrast = 1.035f; exposure = 1.0f
                    detailGain = 0.015f; preserveOriginal = 0.04f
                }
                4, 9, 17, 72 -> { // trees, grass, plants
                    saturation = 1.105f; contrast = 1.055f; exposure = 1.5f
                    detailGain = 0.22f; preserveOriginal = 0.02f
                }
                66 -> { // flowers: richer but not clipped
                    saturation = 1.12f; contrast = 1.025f; exposure = 0.5f
                    detailGain = 0.08f; preserveOriginal = 0.02f
                }
                12 -> { // people: natural skin, softer contrast, no saturation push
                    saturation = 0.985f; contrast = 1.015f; exposure = 1.0f
                    detailGain = 0.025f; preserveOriginal = 0.34f
                }
                92 -> { // clothing: retain fabric color
                    saturation = 1.015f; contrast = 1.025f; exposure = 0.5f
                    detailGain = 0.045f; preserveOriginal = 0.22f
                }
                1, 6 -> { // buildings and roads: recover midtone structure
                    saturation = 1.025f; contrast = 1.075f; exposure = 1.0f
                    detailGain = 0.20f; preserveOriginal = 0.0f
                }
                21 -> { // water
                    saturation = 1.065f; contrast = 1.04f; exposure = 0.5f
                    detailGain = 0.07f; preserveOriginal = 0.02f
                }
                126 -> { // animal: preserve fur/feather color while recovering fine texture
                    saturation = 1.0f; contrast = 1.045f; exposure = 0.5f
                    detailGain = 0.30f; preserveOriginal = 0.12f
                }
                else -> {
                    saturation = 1.035f; contrast = 1.045f; exposure = 0.75f
                    detailGain = 0.11f; preserveOriginal = 0.04f
                }
            }

            // Adaptive shadow lift and highlight roll-off keep dark regions readable
            // while retaining sky/cloud detail. Midtones get the strongest contrast.
            val normalizedY = luma / 255f
            val shadowLift = if (normalizedY < 0.28f) (0.28f - normalizedY) * 5.0f else 0f
            val highlightCompression = if (normalizedY > 0.78f) (normalizedY - 0.78f) * 3.0f else 0f
            val tonedY = (((luma - 127.5f) * contrast) + 127.5f + exposure + shadowLift - highlightCompression)
                .coerceIn(0f, 255f)

            // Local contrast is guided by original detail. Avoid halos on sky, skin and fabric.
            val centerY = sourceY[i]
            val detail = if (w >= 3 && h >= 3) {
                val sx = x.coerceIn(1, w - 2)
                val sy = y.coerceIn(1, h - 2)
                val neighborhood = (sourceY[sy * w + sx - 1] + sourceY[sy * w + sx + 1] +
                    sourceY[(sy - 1) * w + sx] + sourceY[(sy + 1) * w + sx]) * 0.25f
                ((centerY - neighborhood) * detailGain).coerceIn(-14f, 14f)
            } else 0f

            // Scale chroma around the newly toned luminance, preserving hue and
            // preventing a simple saturation-only look.
            val rr0 = tonedY + (r - luma) * saturation + detail
            val gg0 = tonedY + (g - luma) * saturation + detail
            val bb0 = tonedY + (b - luma) * saturation + detail
            val preserve = preserveOriginal
            val rr = rr0 * (1f - preserve) + ((original shr 16) and 255) * preserve
            val gg = gg0 * (1f - preserve) + ((original shr 8) and 255) * preserve
            val bb = bb0 * (1f - preserve) + (original and 255) * preserve
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
        if (small !== source && !small.isRecycled) small.recycle()
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
        val fullMask = Bitmap.createScaledBitmap(lowMask, source.width, source.height, true)
        if (fullMask !== lowMask && !lowMask.isRecycled) lowMask.recycle()
        return fullMask
    }

    override fun close() = interpreter.close()
}
