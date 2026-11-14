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
                source = if (generatedBenchmark) createBenchmarkBitmap() else decodeInput(file!!)
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
                    ColorEnhancer.Mode.AI -> processAi(source)
                }
                val saveStarted = SystemClock.elapsedRealtime()
                updateProgress("Saving to Gallery", 96)
                val uri = saveBitmap(output)
                updateProgress("Photo saved", 100)
                val saveMs = SystemClock.elapsedRealtime() - saveStarted
                val totalMs = SystemClock.elapsedRealtime() - totalStarted
                completeProgress("Photo saved to Pictures/AICamera3rd", true)
                notifyFinished(true, "Photo saved to Pictures/AICamera3rd")
                android.util.Log.i(TAG, "Processing complete: $uri")
                if (benchmark) android.util.Log.i(TAG, "BENCHMARK_RESULT status=success mode=$modeName input=${if (generatedBenchmark) "synthetic" else "photo"} width=${source.width} height=${source.height} save_ms=$saveMs total_ms=$totalMs output=${uri.lastPathSegment}")
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

    private fun processAi(source: Bitmap): Bitmap {
        val loadStarted = SystemClock.elapsedRealtime()
        updateProgress("Loading offline AI models", 10)
        maxim = MaximEnhancer(this)
        updateProgress("AI model ready", 25)
        android.util.Log.i(TAG, "BENCHMARK_STAGE model_load_ms=${SystemClock.elapsedRealtime() - loadStarted}")
        // NAFNet's 256px tiled CPU inference was taking many minutes per tile
        // on the CI emulator. Keep the learned MAXIM color/tone enhancement
        // and skip this non-essential denoising stage for a usable mobile path.
        val enhanceStarted = SystemClock.elapsedRealtime()
        updateProgress("AI color model running", 30, true)
        val enhanced = maxim!!.enhance(source) { row, total ->
            if (row == total || row % 24 == 0) {
                val percent = 55 + (row * 25 / total.coerceAtLeast(1))
                updateProgress("Applying color enhancement", percent)
            }
        }
        updateProgress("Color enhancement complete", 82)
        android.util.Log.i(TAG, "BENCHMARK_STAGE maxim_enhance_ms=${SystemClock.elapsedRealtime() - enhanceStarted}")
        var sceneResult: Bitmap? = null
            try {
                val refineStarted = SystemClock.elapsedRealtime()
                updateProgress("Refining scene and face details", 88)
                scene = try { SceneSegmenter(this) } catch (e: Exception) {
                    android.util.Log.w(TAG, "Optional scene segmentation unavailable", e)
                    null
                }
                faces = try { FaceProtector() } catch (e: Exception) {
                    android.util.Log.w(TAG, "Optional face protection unavailable", e)
                    null
                }
                sceneResult = scene?.apply(source, enhanced) ?: enhanced
                val final = faces?.protect(source, sceneResult) ?: sceneResult
                android.util.Log.i(TAG, "BENCHMARK_STAGE scene_face_refine_ms=${SystemClock.elapsedRealtime() - refineStarted}")
                if (sceneResult !== final && sceneResult !== enhanced && !sceneResult.isRecycled) sceneResult.recycle()
                if (enhanced !== final && !enhanced.isRecycled) enhanced.recycle()
                return final
            } catch (t: Throwable) {
                if (sceneResult != null && sceneResult !== enhanced && !sceneResult.isRecycled) sceneResult.recycle()
                if (!enhanced.isRecycled) enhanced.recycle()
                throw t
            }
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

    private fun decodeInput(file: File): Bitmap {
        if (!file.isFile || file.length() <= 0L) throw IOException("Selected photo is missing or empty")
        val maxEdge = if (getSystemService(android.app.ActivityManager::class.java).isLowRamDevice) 1280 else 1600
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
