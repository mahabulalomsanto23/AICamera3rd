package com.aicamera.pro

import android.graphics.Bitmap
import kotlin.math.pow

/** Deterministic offline color pipeline. This is image processing, not a neural network. */
object ColorEnhancer {
    enum class Mode { AI, VIVID }

    fun enhance(source: Bitmap, mode: Mode): Bitmap {
        val input = source.copy(Bitmap.Config.ARGB_8888, false)
        val w = input.width
        val h = input.height
        val pixels = IntArray(w * h)
        input.getPixels(pixels, 0, w, 0, 0, w, h)
        val vivid = mode == Mode.VIVID
        val sat = if (vivid) 1.34f else 1.19f
        val vibrance = if (vivid) 1.18f else 1.12f
        val contrast = if (vivid) 1.035f else 1.015f
        for (i in pixels.indices) {
            val c = pixels[i]
            var r = ((c shr 16) and 255) / 255f
            var g = ((c shr 8) and 255) / 255f
            var b = (c and 255) / 255f
            // Gentle highlight roll-off and shadow lift retain daylight detail.
            fun tone(v: Float): Float {
                val lifted = (v + 0.018f * (1f - v)) .coerceIn(0f, 1f)
                return (0.5f + (lifted - 0.5f) * contrast).coerceIn(0f, 1f)
            }
            r = tone(r); g = tone(g); b = tone(b)
            val y = 0.2126f*r + 0.7152f*g + 0.0722f*b
            val max = maxOf(r,g,b); val min = minOf(r,g,b)
            val chroma = max - min
            val adaptive = sat + vibrance * (1f - chroma)
            r = (y + (r-y)*adaptive).coerceIn(0f,1f)
            g = (y + (g-y)*adaptive).coerceIn(0f,1f)
            b = (y + (b-y)*adaptive).coerceIn(0f,1f)
            // Subtle warm daylight ambience, weighted down in bright highlights.
            if (!vivid) {
                val warmth = 0.012f * (1f-y)
                r = (r + warmth).coerceIn(0f,1f)
                b = (b - warmth*0.55f).coerceIn(0f,1f)
            }
            pixels[i] = (c and -0x1000000) or
                ((r*255).toInt().coerceIn(0,255) shl 16) or
                ((g*255).toInt().coerceIn(0,255) shl 8) or
                (b*255).toInt().coerceIn(0,255)
        }
        return Bitmap.createBitmap(w,h,Bitmap.Config.ARGB_8888).also { it.setPixels(pixels,0,w,0,0,w,h) }
    }
}
