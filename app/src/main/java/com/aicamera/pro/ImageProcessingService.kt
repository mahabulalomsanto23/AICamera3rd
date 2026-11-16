package com.aicamera.pro

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Debug
import android.os.IBinder
import android.os.PowerManager
import android.os.SystemClock
import android.provider.MediaStore
import androidx.core.app.NotificationCompat
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Owns long-running offline image processing independently of MainActivity.
 * A foreground notification and partial wake lock keep CPU work alive while
 * the UI is minimized or the display is locked.
 */
class ImageProcessingService : Service() {
    private val executor = Executors.newSingleThreadExecutor()
    private var wakeLock: PowerManager.WakeLock? = null
    private var maxim: MaximEnhancer? = null
    private var denoiser: NafNetDenoiser? = null
    private var scene: SceneSegmenter? = null
    private var faces: FaceProtector? = null
    private var selfie: SelfiePersonSegmenter? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        val inputPath = intent?.getStringExtra(EXTRA_INPUT)
        val modeName = intent?.getStringExtra(EXTRA_MODE)
        val strength = intent?.getFloatExtra(EXTRA_STRENGTH, 0.72f) ?: 0.72f
        if (inputPath.isNullOrBlank() || modeName.isNullOrBlank()) {
            stopSelf(startId)
            return START_NOT_STICKY
        }

        startForeground(NOTIFICATION_ID, buildNotification("Preparing photo…", 0, 0, false))
        executor.execute {
            acquireWakeLock()
            val totalStarted = SystemClock.elapsedRealtime()
            val generatedBenchmark = inputPath == "benchmark://generated"
            val benchmark = intent?.getBooleanExtra(EXTRA_BENCHMARK, generatedBenchmark) ?: generatedBenchmark
            var source: Bitmap? = null
            var output: Bitmap? = null
            val file = if (generatedBenchmark) null else File(inputPath)
            try {
                val mode = ColorEnhancer.Mode.valueOf(modeName)
                updateProgress(if (generatedBenchmark) "Creating benchmark photo" else "Opening photo", 2)
                if (mode == ColorEnhancer.Mode.AI && !generatedBenchmark) {
                    // Check system headroom before decoding a large gallery image.
                    ensureAiMemoryHeadroom()
                }
                logMemory("before_decode")
                source = if (generatedBenchmark) createBenchmarkBitmap() else decodeInput(file!!, mode)
                logMemory("after_decode")
                updateProgress("Photo ready", 8)
                output = when (mode) {
                    ColorEnhancer.Mode.VIVID -> {
                        updateProgress("Enhancing vivid colors", 15)
                        ColorEnhancer.enhance(source, ColorEnhancer.Mode.VIVID).also { updateProgress("Color enhancement complete", 82) }
                    }
                    ColorEnhancer.Mode.PORTRAIT -> {
                        updateProgress("Loading portrait segmentation", 12)
                        val segmenter = SelfiePersonSegmenter().also { selfie = it }
                        updateProgress("Detecting portrait subject", 35)
                        val mask = segmenter.personMask(source)
                        try { PortraitProcessor.apply(source, mask, strength).also { updateProgress("Portrait effect complete", 82) } } finally { if (!mask.isRecycled) mask.recycle() }
                    }
                    ColorEnhancer.Mode.AI -> processAi(source, strength)
                }
                if (mode == ColorEnhancer.Mode.AI && output != null) {
                    output = applyMemorySafeDenoise(output!!)
                    output = applyMemorySafeSuperResolution(output!!)
                }
                // MAXIM has produced a separate output bitmap. Drop the decoded
                // source before JPEG encoding so both full-size bitmaps are not
                // retained during the save stage.
                if (mode == ColorEnhancer.Mode.AI && source != null && output !== source) {
                    if (!source!!.isRecycled) source!!.recycle()
                    source = null
                }
                val saveStarted = SystemClock.elapsedRealtime()
                val outputWidth = output.width
                val outputHeight = output.height
                updateProgress("Saving to Gallery", 96)
                val uri = saveBitmap(output)
                // Release the large output bitmap BEFORE notifying the Activity.
                // The completion broadcast restarts CameraX; doing that while the
                // output bitmap is still live can overlap camera buffers with JPEG
                // memory and trigger Android's low-memory process kill.
                if (!output.isRecycled) output.recycle()
                output = null
                System.gc()
                val saveMs = SystemClock.elapsedRealtime() - saveStarted
                val totalMs = SystemClock.elapsedRealtime() - totalStarted
                updateProgress("Photo saved", 100)
                completeProgress("Photo saved to Pictures/AICamera3rd", true)
                notifyFinished(true, "Photo saved to Pictures/AICamera3rd")
                android.util.Log.i(TAG, "Processing complete: $uri")
                if (benchmark) android.util.Log.i(TAG, "BENCHMARK_RESULT status=success mode=$modeName input=${if (generatedBenchmark) "synthetic" else "photo"} width=$outputWidth height=$outputHeight save_ms=$saveMs total_ms=$totalMs output=${uri.lastPathSegment}")
            } catch (t: Throwable) {
                android.util.Log.e(TAG, "Image processing failed", t)
                if (benchmark) android.util.Log.e(TAG, "BENCHMARK_RESULT status=failure elapsed_ms=${SystemClock.elapsedRealtime() - totalStarted} error=${t.javaClass.simpleName}: ${t.message}")
                completeProgress(t.message ?: t.javaClass.simpleName, false)
                notifyFinished(false, t.message ?: t.javaClass.simpleName)
            } finally {
                if (output != null && output !== source && !output.isRecycled) output.recycle()
                if (source != null && !source.isRecycled) source.recycle()
                file?.delete()
                closeModels()
                releaseWakeLock()
                if (stopSelfResult(startId)) {
                    stopForeground(STOP_FOREGROUND_REMOVE)
                }
            }
        }
        return START_REDELIVER_INTENT
    }

    private fun processAi(source: Bitmap, strength: Float): Bitmap {
        val loadStarted = SystemClock.elapsedRealtime()
        logMemory("before_model_load")
        ensureAiMemoryHeadroom()
        updateProgress("Loading offline AI model", 10)
        maxim = MaximEnhancer(this)
        logMemory("after_model_load")
        updateProgress("AI model ready", 25)
        android.util.Log.i(TAG, "BENCHMARK_STAGE model_load_ms=${SystemClock.elapsedRealtime() - loadStarted}")

        // Execute neural stages sequentially: MAXIM -> ADE20K scene-aware tuning -> face preservation.
        // Each optional stage has a memory gate and a fail-open path so the OnePlus 8
        // keeps the last valid image instead of crashing or returning to Home.
        val enhanceStarted = SystemClock.elapsedRealtime()
        updateProgress("AI photo enhancement", 30, true)
        val activeMaxim = maxim ?: throw IllegalStateException("MAXIM model is unavailable")
        var current = try {
            activeMaxim.enhance(source, strength) { row, total ->
                if (row == total || row % 24 == 0) {
                    val percent = 45 + (row * 35 / total.coerceAtLeast(1))
                    updateProgress("AI tone and color analysis", percent)
                }
            }
        } catch (t: Throwable) {
            android.util.Log.e(TAG, "MAXIM enhancement failed", t)
            throw t
        } finally {
            try { activeMaxim.close() } catch (closeError: Exception) {
                android.util.Log.w(TAG, "MAXIM close failed", closeError)
            }
            if (maxim === activeMaxim) maxim = null
        }

        // Recover genuinely dim scenes before semantic grading. The adaptive stage
        // measures luminance first and leaves normally exposed photos untouched.
        try {
            updateProgress("Recovering low-light shadows", 80)
            val lowLightOutput = LowLightEnhancer.enhance(source, strength)
            if (lowLightOutput !== source && lowLightOutput !== current) {
                if (!current.isRecycled) current.recycle()
                current = lowLightOutput
            }
        } catch (t: Throwable) {
            android.util.Log.w(TAG, "LOW_LIGHT_SKIPPED reason=${t.javaClass.simpleName}: ${t.message}", t)
        }

        val memory = android.app.ActivityManager.MemoryInfo()
        (getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager).getMemoryInfo(memory)
        val availableMiB = memory.availMem / (1024L * 1024L)
        if (!memory.lowMemory && availableMiB >= 1800L) {
            var segmenter: SceneSegmenter? = null
            try {
                updateProgress("Understanding scene regions", 82)
                segmenter = SceneSegmenter(this)
                val sceneOutput = segmenter.apply(source, current)
                if (sceneOutput !== current) {
                    if (!current.isRecycled) current.recycle()
                    current = sceneOutput
                }
                android.util.Log.i(TAG, "SCENE_AWARE_DONE classes=ADE20K")
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "SCENE_AWARE_SKIPPED reason=${t.javaClass.simpleName}: ${t.message}", t)
            } finally {
                try { segmenter?.close() } catch (_: Exception) {}
                scene = null
                System.gc()
            }

            var faceProtector: FaceProtector? = null
            try {
                updateProgress("Protecting natural face details", 86)
                faceProtector = FaceProtector()
                val faceOutput = faceProtector.protect(source, current)
                if (faceOutput !== current) {
                    if (!current.isRecycled) current.recycle()
                    current = faceOutput
                }
                android.util.Log.i(TAG, "FACE_PRESERVATION_DONE")
            } catch (t: Throwable) {
                android.util.Log.w(TAG, "FACE_PRESERVATION_SKIPPED reason=${t.javaClass.simpleName}: ${t.message}", t)
            } finally {
                try { faceProtector?.close() } catch (_: Exception) {}
                faces = null
                System.gc()
            }
        } else {
            android.util.Log.i(TAG, "SCENE_FACE_SKIP available_mib=$availableMiB low_memory=${memory.lowMemory}")
        }
        logMemory("after_ai_stages")
        updateProgress("AI enhancement complete", 92)
        android.util.Log.i(TAG, "BENCHMARK_STAGE ai_pipeline_ms=${SystemClock.elapsedRealtime() - enhanceStarted}")
        return current
    }

    /**
     * Run NAFNet on a small working copy, then transfer only its low-frequency
     * denoise correction to the full-size MAXIM result. This keeps source detail
     * and avoids NAFNet's multi-array full-resolution memory cost on OnePlus 8.
     */
    private fun applyMemorySafeDenoise(input: Bitmap): Bitmap {
        val manager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        manager.getMemoryInfo(info)
        val availableMiB = info.availMem / (1024L * 1024L)
        if (info.lowMemory || availableMiB < 2048L) {
            android.util.Log.i(TAG, "NAFNET_SKIP low_memory=true available_mib=$availableMiB")
            return input
        }

        val maxEdge = 768
        val scale = minOf(1.0, maxEdge.toDouble() / maxOf(input.width, input.height))
        val smallWidth = (input.width * scale).toInt().coerceAtLeast(1)
        val smallHeight = (input.height * scale).toInt().coerceAtLeast(1)
        var small: Bitmap? = null
        var denoised: Bitmap? = null
        var active: NafNetDenoiser? = null
        val started = SystemClock.elapsedRealtime()
        try {
            updateProgress("AI noise reduction • preparing", 92)
            small = Bitmap.createScaledBitmap(input, smallWidth, smallHeight, true)
            active = NafNetDenoiser(this)
            denoiser = active
            denoised = active.denoise(small) { done, total ->
                if (done == total || done % 2 == 0) {
                    updateProgress("AI noise reduction • $done/$total", 92 + (done * 3 / total.coerceAtLeast(1)))
                }
            }

            val count = smallWidth * smallHeight
            val originalPixels = IntArray(count)
            val cleanPixels = IntArray(count)
            small.getPixels(originalPixels, 0, smallWidth, 0, 0, smallWidth, smallHeight)
            denoised.getPixels(cleanPixels, 0, smallWidth, 0, 0, smallWidth, smallHeight)

            // Apply a very restrained 12% denoise residual to the original-resolution
            // output. The stronger blend softened fine hair/fur and masonry texture
            // in comparison photos; preserve MAXIM's original high-frequency detail.
            val width = input.width
            val height = input.height
            val baseRow = IntArray(width)
            val outRow = IntArray(width)
            for (y in 0 until height) {
                input.getPixels(baseRow, 0, width, 0, y, width, 1)
                val fy = (y * (smallHeight - 1).toFloat() / maxOf(1, height - 1)).coerceIn(0f, (smallHeight - 1).toFloat())
                val y0 = fy.toInt()
                val y1 = (y0 + 1).coerceAtMost(smallHeight - 1)
                val wy = fy - y0
                for (x in 0 until width) {
                    val fx = (x * (smallWidth - 1).toFloat() / maxOf(1, width - 1)).coerceIn(0f, (smallWidth - 1).toFloat())
                    val x0 = fx.toInt()
                    val x1 = (x0 + 1).coerceAtMost(smallWidth - 1)
                    val wx = fx - x0
                    val i00 = y0 * smallWidth + x0
                    val i10 = y0 * smallWidth + x1
                    val i01 = y1 * smallWidth + x0
                    val i11 = y1 * smallWidth + x1
                    val base = baseRow[x]
                    fun correction(shift: Int): Float {
                        val d00 = ((cleanPixels[i00] shr shift) and 255) - ((originalPixels[i00] shr shift) and 255)
                        val d10 = ((cleanPixels[i10] shr shift) and 255) - ((originalPixels[i10] shr shift) and 255)
                        val d01 = ((cleanPixels[i01] shr shift) and 255) - ((originalPixels[i01] shr shift) and 255)
                        val d11 = ((cleanPixels[i11] shr shift) and 255) - ((originalPixels[i11] shr shift) and 255)
                        val top = d00 * (1f - wx) + d10 * wx
                        val bottom = d01 * (1f - wx) + d11 * wx
                        return (top * (1f - wy) + bottom * wy) * 0.12f
                    }
                    val r = (((base shr 16) and 255) + correction(16)).toInt().coerceIn(0, 255)
                    val g = (((base shr 8) and 255) + correction(8)).toInt().coerceIn(0, 255)
                    val b = ((base and 255) + correction(0)).toInt().coerceIn(0, 255)
                    outRow[x] = (base and -0x1000000) or (r shl 16) or (g shl 8) or b
                }
                input.setPixels(outRow, 0, width, 0, y, width, 1)
            }
            android.util.Log.i(TAG, "NAFNET_DONE size=${smallWidth}x${smallHeight} blend=0.12 elapsed_ms=${SystemClock.elapsedRealtime() - started}")
            return input
        } catch (e: Exception) {
            android.util.Log.w(TAG, "NAFNET_SKIPPED reason=${e.javaClass.simpleName}: ${e.message}", e)
            return input
        } finally {
            try { active?.close() } catch (_: Exception) {}
            if (denoiser === active) denoiser = null
            if (denoised != null && !denoised.isRecycled) denoised.recycle()
            if (small != null && small !== input && !small.isRecycled) small.recycle()
            System.gc()
        }
    }

    /**
     * Run tiled Real-ESRGAN only when memory and source dimensions permit.
     * It creates a true 2x image (4x model output reduced to 2x), not a
     * bitmap resize or edge-sharpening approximation.
     */
    private fun applyMemorySafeSuperResolution(input: Bitmap): Bitmap {
        val manager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        manager.getMemoryInfo(info)
        val availableMiB = info.availMem / (1024L * 1024L)
        val pixels = input.width.toLong() * input.height.toLong()
        if (info.lowMemory || availableMiB < 2500L || pixels > 6_000_000L) {
            android.util.Log.i(TAG, "REAL_ESRGAN_SKIP available_mib=$availableMiB low_memory=${info.lowMemory} pixels=$pixels")
            return input
        }
        var enhancer: RealEsrganEnhancer? = null
        val started = SystemClock.elapsedRealtime()
        try {
            updateProgress("AI super resolution • loading model", 95)
            enhancer = RealEsrganEnhancer(this)
            val result = enhancer.enhance(input) { done, total ->
                if (done == total || done % 4 == 0) {
                    updateProgress("AI super resolution • $done/$total", 95 + (done * 2 / total.coerceAtLeast(1)))
                }
            }
            if (result !== input && !input.isRecycled) input.recycle()
            android.util.Log.i(TAG, "REAL_ESRGAN_DONE elapsed_ms=${SystemClock.elapsedRealtime() - started}")
            return result
        } catch (t: Throwable) {
            android.util.Log.w(TAG, "REAL_ESRGAN_SKIPPED reason=${t.javaClass.simpleName}: ${t.message}", t)
            return input
        } finally {
            try { enhancer?.close() } catch (_: Exception) {}
            System.gc()
        }
    }

    /**
     * MAXIM has a large native tensor working set. Refuse to start when Android
     * reports critically low system memory rather than risking a system-wide LMK.
     */
    private fun ensureAiMemoryHeadroom() {
        val manager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        manager.getMemoryInfo(info)
        // Keep a safety margin without imposing an unnecessarily rigid 1.5 GB
        // free-RAM requirement on every device.
        val required = maxOf(768L * 1024L * 1024L, info.threshold * 3L / 2L)
        android.util.Log.i(TAG, "AI_MEMORY_PREFLIGHT avail=${info.availMem} threshold=${info.threshold} required=$required low=${info.lowMemory}")
        if (info.lowMemory || info.availMem < required) {
            throw IOException("Android reports critically low free memory. Close other apps and try again.")
        }
    }

    /**
     * MAXIM's learned network always runs at 256x256. The source bitmap is used
     * to preserve texture while transferring its learned tone correction, so
     * choose the output working resolution from current system memory.
     */
    private fun selectAiMaxEdge(): Int {
        val manager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        manager.getMemoryInfo(info)
        val availableMiB = info.availMem / (1024L * 1024L)
        // OnePlus 8 has enough headroom for a substantially larger working
        // image. Keep a conservative fallback on devices reporting low memory,
        // but avoid the previous 640-1024px cap that visibly softened output.
        val edge = when {
            manager.isLowRamDevice || availableMiB < 1536L -> 1024
            availableMiB < 2304L -> 1280
            availableMiB < 3072L -> 1536
            else -> 1792
        }
        android.util.Log.i(TAG, "AI_MEMORY_IMAGE_SIZE available_mib=$availableMiB low_ram=${manager.isLowRamDevice} max_edge=$edge")
        return edge
    }

    private fun logMemory(stage: String) {
        val manager = getSystemService(ACTIVITY_SERVICE) as android.app.ActivityManager
        val info = android.app.ActivityManager.MemoryInfo()
        manager.getMemoryInfo(info)
        val runtime = Runtime.getRuntime()
        android.util.Log.i(TAG, "AI_MEMORY stage=$stage avail=${info.availMem} threshold=${info.threshold} low=${info.lowMemory} java_used=${runtime.totalMemory() - runtime.freeMemory()} java_max=${runtime.maxMemory()} native_heap=${Debug.getNativeHeapAllocatedSize()}")
    }

    /**
     * Deterministic 1600x1200 synthetic textured photo for CI performance tests.
     * It exercises the exact offline AI pipeline without requiring a user photo.
     */
    private fun createBenchmarkBitmap(): Bitmap {
        val width = 1600
        val height = 1200
        val pixels = IntArray(width * height)
        var state = 0x13579BDF
        for (y in 0 until height) {
            for (x in 0 until width) {
                state = state * 1664525 + 1013904223
                val noise = (state ushr 24) and 0x3F
                val r = ((x * 255 / width) + noise).coerceIn(0, 255)
                val g = ((y * 220 / height) + noise / 2).coerceIn(0, 255)
                val b = (((x + y) * 180 / (width + height)) + noise).coerceIn(0, 255)
                pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }

    private fun decodeInput(file: File, mode: ColorEnhancer.Mode): Bitmap {
        if (!file.isFile || file.length() <= 0L) throw IOException("Selected photo is missing or empty")
        // Keep the heavy MAXIM path conservative on-device; Vivid/Portrait retain their existing resolution.
        val maxEdge = when {
            mode == ColorEnhancer.Mode.AI -> selectAiMaxEdge()
            getSystemService(android.app.ActivityManager::class.java).isLowRamDevice -> 1280
            else -> 1600
        }
        if (Build.VERSION.SDK_INT >= 28) {
            val source = ImageDecoder.createSource(file)
            return ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                val scale = minOf(1.0, maxEdge.toDouble() / maxOf(info.size.width, info.size.height))
                decoder.setTargetSize(
                    (info.size.width * scale).toInt().coerceAtLeast(1),
                    (info.size.height * scale).toInt().coerceAtLeast(1)
                )
                decoder.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            }
        }
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("Unsupported or damaged photo")
        var sample = 1
        while (maxOf(bounds.outWidth / sample, bounds.outHeight / sample) > maxEdge) sample *= 2
        return BitmapFactory.decodeFile(file.absolutePath, BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }) ?: throw IOException("Could not decode selected photo")
    }

    private fun saveBitmap(bitmap: Bitmap): Uri {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "AICamera3rd_$stamp.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AICamera3rd")
            if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Images.Media.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("Android could not create the Gallery file")
        try {
            contentResolver.openOutputStream(uri)?.use { stream ->
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 96, stream)) throw IOException("JPEG encoding failed")
                stream.flush()
            } ?: throw IOException("Could not open Gallery output stream")
            if (Build.VERSION.SDK_INT >= 29) {
                contentResolver.update(uri, ContentValues().apply {
                    put(MediaStore.Images.Media.IS_PENDING, 0)
                }, null, null)
            }
            return uri
        } catch (t: Throwable) {
            contentResolver.delete(uri, null, null)
            throw t
        }
    }

    private fun acquireWakeLock() {
        val manager = getSystemService(POWER_SERVICE) as PowerManager
        wakeLock = manager.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$TAG:ImageProcessing").apply {
            setReferenceCounted(false)
            acquire(2 * 60 * 60 * 1000L)
        }
    }

    private fun releaseWakeLock() {
        try { if (wakeLock?.isHeld == true) wakeLock?.release() } catch (_: Exception) {}
        wakeLock = null
    }

    private fun updateProgress(title: String, percent: Int, indeterminate: Boolean = false) {
        val safePercent = percent.coerceIn(0, 100)
        getSharedPreferences(PROGRESS_PREFS, MODE_PRIVATE).edit()
            .putBoolean("active", true).putString("title", title).putInt("percent", safePercent).apply()
        sendProgressBroadcast(title, safePercent, true)
        val notification = buildNotification(title, safePercent, 100, true, indeterminate)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIFICATION_ID, notification)
    }

    private fun completeProgress(message: String, success: Boolean) {
        val title = if (success) "Saved to Gallery" else "Processing failed"
        getSharedPreferences(PROGRESS_PREFS, MODE_PRIVATE).edit()
            .putBoolean("active", false).putString("title", title).putInt("percent", if (success) 100 else 0).apply()
        sendProgressBroadcast(title, if (success) 100 else 0, false)
    }

    private fun sendProgressBroadcast(title: String, percent: Int, active: Boolean) {
        sendBroadcast(Intent(ACTION_PROGRESS).setPackage(packageName).apply {
            putExtra(EXTRA_PROGRESS_TITLE, title)
            putExtra(EXTRA_PROGRESS_PERCENT, percent)
            putExtra(EXTRA_PROGRESS_ACTIVE, active)
        })
    }

    private fun buildNotification(title: String, done: Int, total: Int, ongoing: Boolean, indeterminate: Boolean = false): Notification {
        val openApp = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        val builder = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_menu_camera)
            .setContentTitle("AI Camera Pro")
            .setContentText(title)
            .setContentIntent(openApp)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
            .setPriority(NotificationCompat.PRIORITY_LOW)
        if (indeterminate && ongoing) builder.setProgress(0, 0, true)
        else if (total > 0) builder.setProgress(total, done, false)
        return builder.build()
    }

    private fun notifyFinished(success: Boolean, message: String) {
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(if (success) android.R.drawable.ic_menu_save else android.R.drawable.ic_dialog_alert)
            .setContentTitle(if (success) "AI Camera Pro • Saved" else "AI Camera Pro • Processing failed")
            .setContentText(message)
            .setStyle(NotificationCompat.BigTextStyle().bigText(message))
            .setAutoCancel(true)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .build()
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(RESULT_NOTIFICATION_ID, notification)
    }

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= 26) {
            val channel = NotificationChannel(CHANNEL_ID, "Photo processing", NotificationManager.IMPORTANCE_LOW)
            channel.description = "Shows progress while AI Camera Pro processes photos"
            (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(channel)
        }
    }

    private fun closeModels() {
        try { faces?.close() } catch (_: Exception) {}
        try { scene?.close() } catch (_: Exception) {}
        try { denoiser?.close() } catch (_: Exception) {}
        try { maxim?.close() } catch (_: Exception) {}
        try { selfie?.close() } catch (_: Exception) {}
        faces = null; scene = null; denoiser = null; maxim = null; selfie = null
    }

    override fun onDestroy() {
        releaseWakeLock()
        executor.shutdown()
        closeModels()
        super.onDestroy()
    }

    companion object {
        const val EXTRA_INPUT = "com.aicamera.pro.extra.INPUT"
        const val EXTRA_MODE = "com.aicamera.pro.extra.MODE"
        const val EXTRA_STRENGTH = "com.aicamera.pro.extra.STRENGTH"
        const val EXTRA_BENCHMARK = "com.aicamera.pro.extra.BENCHMARK"
        const val ACTION_PROGRESS = "com.aicamera.pro.action.PROCESSING_PROGRESS"
        const val EXTRA_PROGRESS_TITLE = "progress_title"
        const val EXTRA_PROGRESS_PERCENT = "progress_percent"
        const val EXTRA_PROGRESS_ACTIVE = "progress_active"
        const val PROGRESS_PREFS = "processing_progress"
        private const val TAG = "AICameraProcessing"
        private const val CHANNEL_ID = "ai_photo_processing"
        private const val NOTIFICATION_ID = 4101
        private const val RESULT_NOTIFICATION_ID = 4102
    }
}
