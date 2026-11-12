package com.aicamera.pro

import android.graphics.Bitmap
import kotlin.math.max

/** Offline portrait depth effect with subject-preserving compositing. */
object PortraitProcessor {
    fun apply(source: Bitmap, personMask: Bitmap, strength: Float): Bitmap {
        require(personMask.width == source.width && personMask.height == source.height) {
            "Portrait mask size does not match the photo"
        }
        val w = source.width
        val h = source.height
        val sw = max(1, w / 4)
        val sh = max(1, h / 4)
        val small = Bitmap.createScaledBitmap(source, sw, sh, true)
        var bgPixels = IntArray(sw * sh)
        small.getPixels(bgPixels, 0, sw, 0, 0, sw, sh)
        val radius = when {
            strength < 0.6f -> 3
            strength < 0.8f -> 5
            else -> 8
        }
        repeat(3) { bgPixels = boxBlur(bgPixels, sw, sh, radius) }
        val blurredSmall = Bitmap.createBitmap(sw, sh, Bitmap.Config.ARGB_8888)
        blurredSmall.setPixels(bgPixels, 0, sw, 0, 0, sw, sh)
        val blurred = Bitmap.createScaledBitmap(blurredSmall, w, h, true)

        val fgPixels = IntArray(w * h)
        val bgFull = IntArray(w * h)
        val maskPixels = IntArray(w * h)
        source.getPixels(fgPixels, 0, w, 0, 0, w, h)
        blurred.getPixels(bgFull, 0, w, 0, 0, w, h)
        personMask.getPixels(maskPixels, 0, w, 0, 0, w, h)
        val result = IntArray(w * h)
        for (i in result.indices) {
            val raw = (maskPixels[i] and 255) / 255f
            val t = ((raw - 0.10f) / 0.80f).coerceIn(0f, 1f)
            val subject = t * t * (3f - 2f * t)
            val inv = 1f - subject
            val fg = fgPixels[i]
            val bg = bgFull[i]
            val a = (fg ushr 24) and 255
            val r = (((fg shr 16) and 255) * subject + ((bg shr 16) and 255) * inv).toInt().coerceIn(0,255)
            val g = (((fg shr 8) and 255) * subject + ((bg shr 8) and 255) * inv).toInt().coerceIn(0,255)
            val b = ((fg and 255) * subject + (bg and 255) * inv).toInt().coerceIn(0,255)
            result[i] = (a shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(result, w, h, Bitmap.Config.ARGB_8888)
    }

    /** Three separable sliding-window passes approximate a Gaussian background blur. */
    private fun boxBlur(input: IntArray, w: Int, h: Int, radius: Int): IntArray {
        val horizontal = IntArray(input.size)
        val output = IntArray(input.size)
        val window = radius * 2 + 1
        for (y in 0 until h) {
            var a=0; var r=0; var g=0; var b=0
            for (k in -radius..radius) {
                val p=input[y*w+k.coerceIn(0,w-1)]
                a+=(p ushr 24) and 255; r+=(p shr 16) and 255
                g+=(p shr 8) and 255; b+=p and 255
            }
            for (x in 0 until w) {
                horizontal[y*w+x]=((a/window) shl 24) or ((r/window) shl 16) or ((g/window) shl 8) or (b/window)
                val rem=input[y*w+(x-radius).coerceIn(0,w-1)]
                val add=input[y*w+(x+radius+1).coerceIn(0,w-1)]
                a+=((add ushr 24) and 255)-((rem ushr 24) and 255)
                r+=((add shr 16) and 255)-((rem shr 16) and 255)
                g+=((add shr 8) and 255)-((rem shr 8) and 255)
                b+=(add and 255)-(rem and 255)
            }
        }
        for (x in 0 until w) {
            var a=0; var r=0; var g=0; var b=0
            for (k in -radius..radius) {
                val p=horizontal[k.coerceIn(0,h-1)*w+x]
                a+=(p ushr 24) and 255; r+=(p shr 16) and 255
                g+=(p shr 8) and 255; b+=p and 255
            }
            for (y in 0 until h) {
                output[y*w+x]=((a/window) shl 24) or ((r/window) shl 16) or ((g/window) shl 8) or (b/window)
                val rem=horizontal[(y-radius).coerceIn(0,h-1)*w+x]
                val add=horizontal[(y+radius+1).coerceIn(0,h-1)*w+x]
                a+=((add ushr 24) and 255)-((rem ushr 24) and 255)
                r+=((add shr 16) and 255)-((rem shr 16) and 255)
                g+=((add shr 8) and 255)-((rem shr 8) and 255)
                b+=(add and 255)-(rem and 255)
            }
        }
        return output
    }
}
