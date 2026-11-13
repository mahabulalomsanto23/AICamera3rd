package com.aicamera.pro

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.File

/**
 * Debug-only entry point for CI to exercise the real foreground image-processing service.
 * It only accepts the fixed test image inside this app's external files directory.
 */
class BenchmarkReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_RUN) return

        val suppliedPath = intent.getStringExtra(ImageProcessingService.EXTRA_INPUT) ?: return
        val expectedFile = File(
            context.getExternalFilesDir(null),
            "aicamera3rd-test-photo.jpg"
        ).canonicalFile
        val suppliedFile = File(suppliedPath).canonicalFile
        if (suppliedFile != expectedFile || !suppliedFile.isFile) {
            Log.e(TAG, "Rejected CI benchmark input path")
            return
        }

        val mode = intent.getStringExtra(ImageProcessingService.EXTRA_MODE)
            ?: ColorEnhancer.Mode.AI.name
        val serviceIntent = Intent(context, ImageProcessingService::class.java).apply {
            putExtra(ImageProcessingService.EXTRA_INPUT, suppliedFile.absolutePath)
            putExtra(ImageProcessingService.EXTRA_MODE, mode)
            putExtra(ImageProcessingService.EXTRA_BENCHMARK, true)
        }
        ContextCompat.startForegroundService(context, serviceIntent)
        Log.i(TAG, "Started CI photo benchmark")
    }

    companion object {
        const val ACTION_RUN = "com.aicamera.pro.RUN_BENCHMARK"
        private const val TAG = "AICameraBenchmark"
    }
}
