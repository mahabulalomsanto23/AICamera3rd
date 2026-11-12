package com.aicamera.pro

import android.graphics.Bitmap
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow

/**
 * Offline adaptive color pipeline. It is image processing, not a neural network.
 * AI mode favors natural daylight color; VIVID mode applies a stronger creative look.
 */
object ColorEnhancer {
    enum class Mode { AI, VIVID, PORTRAIT }

    fun enhance(source: Bitmap, mode: Mode): Bitmap {
        val input = if (source.config == Bitmap.Config.ARGB_8888) source else source.copy(Bitmap.Config.ARGB_8888, false)
        val w = input.width
        val h = input.height
        val pixels = IntArray(w * h)
        input.getPixels(pixels, 0, w, 0, 0, w, h)

        // Estimate a restrained scene white balance from a sparse sample.
        var sr = 0.0; var sg = 0.0; var sb = 0.0; var samples = 0
        val step = max(1, pixels.size / 12000)
        for (i in pixels.indices step step) {
            val c = pixels[i]
            sr += (c shr 16 and 255) / 255.0
            sg += (c shr 8 and 255) / 255.0
            sb += (c and 255) / 255.0
            samples++
        }
        val ar = sr / samples; val ag = sg / samples; val ab = sb / samples
        val neutral = ((ar + ag + ab) / 3.0).coerceAtLeast(0.001)
        val balanceStrength = if (mode == Mode.AI) 0.16 else 0.06
        val wr = (1.0 + ((neutral / ar.coerceAtLeast(0.001)) - 1.0) * balanceStrength).coerceIn(0.94, 1.06).toFloat()
        val wg = (1.0 + ((neutral / ag.coerceAtLeast(0.001)) - 1.0) * balanceStrength).coerceIn(0.94, 1.06).toFloat()
        val wb = (1.0 + ((neutral / ab.coerceAtLeast(0.001)) - 1.0) * balanceStrength).coerceIn(0.94, 1.06).toFloat()

        val vivid = mode == Mode.VIVID
        val saturationBase = if (vivid) 1.48f else 1.30f
        val vibrance = if (vivid) 0.30f else 0.24f
        val contrast = if (vivid) 1.055f else 1.035f
        val warmth = if (vivid) 0.002f else 0.006f

        for (i in pixels.indices) {
            val c = pixels[i]
            var r = ((c shr 16) and 255) / 255f * wr
            var g = ((c shr 8) and 255) / 255f * wg
            var b = (c and 255) / 255f * wb
            r = r.coerceIn(0f, 1f); g = g.coerceIn(0f, 1f); b = b.coerceIn(0f, 1f)

            // Gentle shadow lift, midtone contrast and highlight protection.
            fun tone(v: Float): Float {
                val lifted = (v + 0.012f * (1f - v) * (1f - v)).coerceIn(0f, 1f)
                val contrasted = (0.5f + (lifted - 0.5f) * contrast).coerceIn(0f, 1f)
                return if (contrasted > 0.78f) {
                    0.78f + (contrasted - 0.78f) * 0.88f
                } else contrasted
            }
            r = tone(r); g = tone(g); b = tone(b)

            val y = (0.2126f * r + 0.7152f * g + 0.0722f * b).coerceIn(0f, 1f)
            val mx = max(r, max(g, b))
            val mn = min(r, min(g, b))
            val chroma = mx - mn
            // Vibrance favors muted colors rather than pushing already vivid pixels.
            var sat = saturationBase + vibrance * (1f - chroma)
            // Approximate skin-tone protection: avoid over-saturating warm orange/red hues.
            val skinLike = r > g * 1.08f && g > b * 1.12f && r > 0.28f && y in 0.18f..0.88f
            if (skinLike) sat *= if (vivid) 0.88f else 0.84f
            // Keep very dark and very bright areas from becoming unnaturally colorful.
            val tonalGuard = (0.82f + 0.18f * (1f - kotlin.math.abs(y - 0.5f) * 2f)).coerceIn(0.82f, 1f)
            sat = 1f + (sat - 1f) * tonalGuard

            r = (y + (r - y) * sat).coerceIn(0f, 1f)
            g = (y + (g - y) * sat).coerceIn(0f, 1f)
            b = (y + (b - y) * sat).coerceIn(0f, 1f)

            // Subtle daylight warmth only in midtones, without yellowing highlights.
            val midtone = (1f - kotlin.math.abs(y - 0.52f) * 1.8f).coerceIn(0f, 1f)
            r = (r + warmth * midtone).coerceIn(0f, 1f)
            b = (b - warmth * 0.45f * midtone).coerceIn(0f, 1f)

            pixels[i] = (c and -0x1000000) or
                ((r * 255f).toInt().coerceIn(0, 255) shl 16) or
                ((g * 255f).toInt().coerceIn(0, 255) shl 8) or
                (b * 255f).toInt().coerceIn(0, 255)
        }

        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).also {
            it.setPixels(pixels, 0, w, 0, 0, w, h)
        }
    }
}
