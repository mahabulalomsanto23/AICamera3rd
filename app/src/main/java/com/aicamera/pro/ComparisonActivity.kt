package com.aicamera.pro

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.os.Bundle
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.SeekBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.File
import kotlin.math.min

/** Full-screen, on-device comparison of the source and enhanced result. */
class ComparisonActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val originalPath = intent.getStringExtra(EXTRA_ORIGINAL)
        val enhancedPath = intent.getStringExtra(EXTRA_ENHANCED)
        if (originalPath.isNullOrBlank() || enhancedPath.isNullOrBlank()) {
            finish()
            return
        }
        val originalFile = File(originalPath)
        val enhancedFile = File(enhancedPath)
        if (!originalFile.isFile || !enhancedFile.isFile) {
            finish()
            return
        }
        val original = decodePreview(originalFile)
        val enhanced = decodePreview(enhancedFile)
        if (original == null || enhanced == null) {
            original?.recycle()
            enhanced?.recycle()
            finish()
            return
        }

        window.statusBarColor = Color.BLACK
        window.navigationBarColor = Color.BLACK
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.BLACK)
            setPadding(dp(12), dp(12), dp(12), dp(16))
        }
        val title = TextView(this).apply {
            text = "BEFORE  /  AFTER"
            setTextColor(Color.WHITE)
            textSize = 15f
            gravity = Gravity.CENTER
            setTypeface(null, android.graphics.Typeface.BOLD)
            letterSpacing = 0.08f
        }
        root.addView(title, LinearLayout.LayoutParams(-1, dp(48)))
        val compare = ComparisonView(this, original, enhanced)
        root.addView(compare, LinearLayout.LayoutParams(-1, 0, 1f).apply {
            topMargin = dp(8)
            bottomMargin = dp(14)
        })
        val hint = TextView(this).apply {
            text = "DRAG THE DIVIDER TO COMPARE"
            setTextColor(0xFFCCCCCC.toInt())
            textSize = 11f
            gravity = Gravity.CENTER
        }
        root.addView(hint, LinearLayout.LayoutParams(-1, dp(30)))
        val slider = SeekBar(this).apply {
            max = 100
            progress = 50
            contentDescription = "Before and after comparison position"
            setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
                override fun onProgressChanged(seekBar: SeekBar?, progress: Int, fromUser: Boolean) {
                    compare.setReveal(progress / 100f)
                }
                override fun onStartTrackingTouch(seekBar: SeekBar?) {}
                override fun onStopTrackingTouch(seekBar: SeekBar?) {}
            })
        }
        root.addView(slider, LinearLayout.LayoutParams(-1, dp(42)))
        val close = TextView(this).apply {
            text = "DONE"
            setTextColor(Color.WHITE)
            textSize = 13f
            gravity = Gravity.CENTER
            setTypeface(null, android.graphics.Typeface.BOLD)
            setOnClickListener { finish() }
        }
        root.addView(close, LinearLayout.LayoutParams(-1, dp(44)))
        setContentView(root)
    }

    private fun decodePreview(file: File): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (maxOf(bounds.outWidth / sample, bounds.outHeight / sample) > 1800) sample *= 2
        return BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        })
    }

    private fun dp(value: Int) = (value * resources.displayMetrics.density + 0.5f).toInt()

    companion object {
        const val EXTRA_ORIGINAL = "comparison.original"
        const val EXTRA_ENHANCED = "comparison.enhanced"
    }
}

private class ComparisonView(
    context: android.content.Context,
    private val original: Bitmap,
    private val enhanced: Bitmap
) : View(context) {
    private val imageRect = RectF()
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG)
    private val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        strokeWidth = 3f * resources.displayMetrics.density
    }
    private val labelPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 11f * resources.displayMetrics.density
        typeface = android.graphics.Typeface.create(android.graphics.Typeface.DEFAULT, android.graphics.Typeface.BOLD)
    }
    private var reveal = 0.5f

    fun setReveal(value: Float) {
        reveal = value.coerceIn(0.02f, 0.98f)
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(0xFF101010.toInt())
        if (imageRect.isEmpty) {
            val scale = min(width.toFloat() / original.width, height.toFloat() / original.height)
            val drawW = original.width * scale
            val drawH = original.height * scale
            imageRect.set((width - drawW) / 2f, (height - drawH) / 2f,
                (width + drawW) / 2f, (height + drawH) / 2f)
        }
        canvas.drawBitmap(enhanced, null, imageRect, paint)
        val split = imageRect.left + imageRect.width() * reveal
        val save = canvas.save()
        canvas.clipRect(imageRect.left, imageRect.top, split, imageRect.bottom)
        canvas.drawBitmap(original, null, imageRect, paint)
        canvas.restoreToCount(save)
        canvas.drawLine(split, imageRect.top, split, imageRect.bottom, dividerPaint)
        canvas.drawCircle(split, imageRect.centerY(), 16f * resources.displayMetrics.density, dividerPaint)
        drawLabel(canvas, "ORIGINAL", imageRect.left + 12f, imageRect.top + 26f)
        drawLabel(canvas, "AI ENHANCED", imageRect.right - 112f * resources.displayMetrics.density, imageRect.top + 26f)
    }

    private fun drawLabel(canvas: Canvas, text: String, x: Float, y: Float) {
        val pad = 8f * resources.displayMetrics.density
        val width = labelPaint.measureText(text)
        val box = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = 0xAA000000.toInt() }
        canvas.drawRoundRect(x - pad, y - 17f * resources.displayMetrics.density,
            x + width + pad, y + 7f * resources.displayMetrics.density,
            6f, 6f, box)
        canvas.drawText(text, x, y, labelPaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.action) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                if (imageRect.width() > 0f) {
                    setReveal((event.x - imageRect.left) / imageRect.width())
                    parent?.requestDisallowInterceptTouchEvent(true)
                }
                return true
            }
            MotionEvent.ACTION_UP -> {
                performClick()
                return true
            }
        }
        return true
    }

    override fun performClick(): Boolean {
        super.performClick()
        return true
    }
}
