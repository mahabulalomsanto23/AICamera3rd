package com.aicamera.pro

import android.graphics.Bitmap
import kotlin.math.pow

/**
 * Adaptive low-light recovery for dim photographs.
 * Uses luminance statistics to lift shadows while protecting highlights and
 * retaining chroma. This is a deterministic ISP-style stage, not a neural model.
 */
object LowLightEnhancer {
    data class Analysis(val meanLuma: Float, val darkRatio: Float, val gain: Float)

    fun analyze(source: Bitmap): Analysis {
        val w = source.width
        val h = source.height
        val stepX = maxOf(1, w / 96)
        val stepY = maxOf(1, h / 96)
        var sum = 0.0
        var dark = 0
        var count = 0
        for (y in 0 until h step stepY) {
            for (x in 0 until w step stepX) {
                val p = source.getPixel(x, y)
                val l = (0.2126 * ((p shr 16) and 255) +
                    0.7152 * ((p shr 8) and 255) + 0.0722 * (p and 255)) / 255.0
                sum += l
                if (l < 0.22) dark++
                count++
            }
        }
        val mean = (sum / count.coerceAtLeast(1)).toFloat()
        val darkRatio = dark.toFloat() / count.coerceAtLeast(1)
        val gain = when {
            mean < 0.12f -> 1.42f
            mean < 0.20f -> 1.30f
            mean < 0.29f -> 1.18f
            mean < 0.38f -> 1.08f
            else -> 1f
        }
        return Analysis(mean, darkRatio, gain)
    }

    fun enhance(source: Bitmap, strength: Float = 0.72f): Bitmap {
        val analysis = analyze(source)
        if (analysis.gain <= 1.015f || analysis.darkRatio < 0.18f) return source
        val amount = strength.coerceIn(0f, 1f)
        val gain = 1f + (analysis.gain - 1f) * amount
        val gamma = (1f - 0.16f * amount * (analysis.darkRatio.coerceIn(0f, 1f))).coerceIn(0.78f, 1f)
        val w = source.width
        val h = source.height
        val row = IntArray(w)
        val result = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        try {
            for (y in 0 until h) {
                source.getPixels(row, 0, w, 0, y, w, 1)
                for (x in row.indices) {
                    val p = row[x]
                    val r = (p shr 16) and 255
                    val g = (p shr 8) and 255
                    val b = p and 255
                    val l = (0.2126f * r + 0.7152f * g + 0.0722f * b) / 255f
                    val lifted = (l * gain).coerceAtMost(1f).toDouble().pow(gamma.toDouble()).toFloat()
                    // Preserve hue by scaling RGB together; smoothly reduce lift in highlights.
                    val highlightGuard = (1f - ((l - 0.62f) / 0.38f).coerceIn(0f, 1f) * 0.72f)
                    val targetL = l + (lifted - l) * highlightGuard
                    val scale = if (l > 0.015f) targetL / l else 1f + (gain - 1f) * 0.35f
                    val rr = (r * scale).toInt().coerceIn(0, 255)
                    val gg = (g * scale).toInt().coerceIn(0, 255)
                    val bb = (b * scale).toInt().coerceIn(0, 255)
                    row[x] = (p and -0x1000000) or (rr shl 16) or (gg shl 8) or bb
                }
                result.setPixels(row, 0, w, 0, y, w, 1)
            }
            android.util.Log.i("AICameraLowLight", "LOW_LIGHT_APPLIED mean=${analysis.meanLuma} dark_ratio=${analysis.darkRatio} gain=$gain strength=$amount")
            return result
        } catch (t: Throwable) {
            if (!result.isRecycled) result.recycle()
            throw t
        }
    }
}
