package com.aicamera.pro

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.max
import kotlin.math.min

/** NAFNet-SIDD width-32 local denoiser. Uses overlapping 256px tiles. */
class NafNetDenoiser(context: Context) : AutoCloseable {
    private val interpreter: Interpreter

    init {
        val file = File(context.filesDir, "nafnet_sidd_width32_fp16.tflite")
        if (!file.exists() || file.length() == 0L) {
            context.assets.open("nafnet_sidd_width32_fp16.tflite").use { input ->
                FileOutputStream(file).use { input.copyTo(it) }
            }
        }
        // Use the CPU backend for device compatibility. Creating the Adreno GPU
        // delegate during initialization can terminate some vendor-driver processes
        // with a native signal that Java exception handlers cannot catch.
        interpreter = Interpreter(file, Interpreter.Options().apply { setNumThreads(4) })
        require(interpreter.getInputTensor(0).shape().contentEquals(intArrayOf(1, 3, 256, 256))) {
            "Unexpected NAFNet input: ${interpreter.getInputTensor(0).shape().contentToString()}"
        }
        require(interpreter.getOutputTensor(0).shape().contentEquals(intArrayOf(1, 3, 256, 256))) {
            "Unexpected NAFNet output: ${interpreter.getOutputTensor(0).shape().contentToString()}"
        }
    }

    fun denoise(source: Bitmap, onProgress: ((Int, Int) -> Unit)? = null): Bitmap {
        val w = source.width
        val h = source.height
        val pixels = IntArray(w * h)
        source.getPixels(pixels, 0, w, 0, 0, w, h)
        val result = IntArray(pixels.size)
        val weight = FloatArray(pixels.size)
        val tile = 256
        val stride = 224
        val input = ByteBuffer.allocateDirect(3 * tile * tile * 4).order(ByteOrder.nativeOrder())
        val output = ByteBuffer.allocateDirect(3 * tile * tile * 4).order(ByteOrder.nativeOrder())
        val patch = IntArray(tile * tile)
        val planes = Array(3) { FloatArray(tile * tile) }
        val totalTiles = ((h + stride - 1) / stride) * ((w + stride - 1) / stride)
        var completedTiles = 0
        var top = 0
        while (top < h) {
            var left = 0
            while (left < w) {
                input.clear()
                output.clear()
                for (y in 0 until tile) for (x in 0 until tile) {
                    val sx = (left + x - 16).coerceIn(0, w - 1)
                    val sy = (top + y - 16).coerceIn(0, h - 1)
                    patch[y * tile + x] = pixels[sy * w + sx]
                }
                for (channel in 0..2) for (i in patch.indices) {
                    val c = patch[i]
                    val v = when (channel) { 0 -> c shr 16 and 255; 1 -> c shr 8 and 255; else -> c and 255 }
                    input.putFloat(v / 255f)
                }
                input.rewind()
                interpreter.run(input, output)
                output.rewind()
                for (channel in 0..2) for (i in 0 until tile * tile) planes[channel][i] = output.float.coerceIn(0f, 1f)
                val endY = min(h, top + stride)
                val endX = min(w, left + stride)
                for (y in top until endY) for (x in left until endX) {
                    val px = x - left + 16
                    val py = y - top + 16
                    val pi = py * tile + px
                    val idx = y * w + x
                    val edgeX = min(min(px, tile - 1 - px), 16).coerceAtLeast(0) / 16f
                    val edgeY = min(min(py, tile - 1 - py), 16).coerceAtLeast(0) / 16f
                    val blend = (0.35f + 0.65f * min(edgeX, edgeY)).coerceIn(0.35f, 1f)
                    val rr = (planes[0][pi] * 255f).toInt().coerceIn(0,255)
                    val gg = (planes[1][pi] * 255f).toInt().coerceIn(0,255)
                    val bb = (planes[2][pi] * 255f).toInt().coerceIn(0,255)
                    val den = (0xFF shl 24) or (rr shl 16) or (gg shl 8) or bb
                    val oldW = weight[idx]
                    val addW = blend
                    val old = result[idx]
                    val nr = (((old shr 16 and 255) * oldW + rr * addW) / (oldW + addW)).toInt()
                    val ng = (((old shr 8 and 255) * oldW + gg * addW) / (oldW + addW)).toInt()
                    val nb = (((old and 255) * oldW + bb * addW) / (oldW + addW)).toInt()
                    result[idx] = (0xFF shl 24) or (nr shl 16) or (ng shl 8) or nb
                    weight[idx] = oldW + addW
                }
                completedTiles++
                onProgress?.invoke(completedTiles, totalTiles)
                left += stride
            }
            top += stride
        }
        return Bitmap.createBitmap(result, w, h, Bitmap.Config.ARGB_8888)
    }

    override fun close() {
        interpreter.close()
    }
}
