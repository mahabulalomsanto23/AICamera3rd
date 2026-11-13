package com.aicamera.pro

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.PowerManager
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
            var source: Bitmap? = null
            var output: Bitmap? = null
            val file = File(inputPath)
            try {
                val mode = ColorEnhancer.Mode.valueOf(modeName)
                updateProgress("Opening photo", 0, 0)
                source = decodeInput(file)
                output = when (mode) {
                    ColorEnhancer.Mode.VIVID -> {
                        updateProgress("Enhancing vivid colors", 0, 0)
                        ColorEnhancer.enhance(source, ColorEnhancer.Mode.VIVID)
                    }
                    ColorEnhancer.Mode.PORTRAIT -> {
                        updateProgress("Loading portrait segmentation", 0, 0)
                        val segmenter = SelfiePersonSegmenter().also { selfie = it }
                        val mask = segmenter.personMask(source)
                        try { PortraitProcessor.apply(source, mask, strength) } finally { if (!mask.isRecycled) mask.recycle() }
                    }
                    ColorEnhancer.Mode.AI -> processAi(source)
                }
                updateProgress("Saving to Gallery", 0, 0)
                val uri = saveBitmap(output)
                notifyFinished(true, "Photo saved to Pictures/AICamera3rd")
                android.util.Log.i(TAG, "Processing complete: $uri")
            } catch (t: Throwable) {
                android.util.Log.e(TAG, "Image processing failed", t)
                notifyFinished(false, t.message ?: t.javaClass.simpleName)
            } finally {
                if (output != null && output !== source && !output.isRecycled) output.recycle()
                if (source != null && !source.isRecycled) source.recycle()
                file.delete()
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
        updateProgress("Loading offline AI models", 0, 0)
        maxim = MaximEnhancer(this)
        denoiser = NafNetDenoiser(this)
        updateProgress("Denoising details", 0, 0)
        val denoised = denoiser!!.denoise(source) { done, total ->
            if (done == 1 || done == total || done % 2 == 0) {
                updateProgress("Denoising details", done, total)
            }
        }
        try {
            updateProgress("Enhancing colors", 0, 0)
            val enhanced = maxim!!.enhance(denoised)
            if (!denoised.isRecycled) denoised.recycle()
            var sceneResult: Bitmap? = null
            try {
                updateProgress("Refining scene and face details", 0, 0)
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
                if (sceneResult !== final && sceneResult !== enhanced && !sceneResult.isRecycled) sceneResult.recycle()
                if (enhanced !== final && !enhanced.isRecycled) enhanced.recycle()
                return final
            } catch (t: Throwable) {
                if (sceneResult != null && sceneResult !== enhanced && !sceneResult.isRecycled) sceneResult.recycle()
                if (!enhanced.isRecycled) enhanced.recycle()
                throw t
            }
        } finally {
            if (!denoised.isRecycled) denoised.recycle()
        }
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

    private fun updateProgress(title: String, done: Int, total: Int) {
        val notification = buildNotification(title, done, total, true)
        (getSystemService(NOTIFICATION_SERVICE) as NotificationManager).notify(NOTIFICATION_ID, notification)
    }

    private fun buildNotification(title: String, done: Int, total: Int, ongoing: Boolean): Notification {
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
        if (total > 0) builder.setProgress(total, done, false)
        else if (ongoing) builder.setProgress(0, 0, true)
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
        private const val TAG = "AICameraProcessing"
        private const val CHANNEL_ID = "ai_photo_processing"
        private const val NOTIFICATION_ID = 4101
        private const val RESULT_NOTIFICATION_ID = 4102
    }
}
