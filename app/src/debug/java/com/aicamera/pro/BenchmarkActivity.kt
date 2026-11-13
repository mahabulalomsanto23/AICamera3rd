package com.aicamera.pro

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat
import java.io.File

/**
 * Debug-only CI entry point. The actual processing service remains private;
 * this exported activity validates the fixed test file and starts it in-app.
 */
class BenchmarkActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            val suppliedPath = intent.getStringExtra(ImageProcessingService.EXTRA_INPUT)
                ?: throw IllegalArgumentException("Missing benchmark input path")
            val expectedFile = File(
                getExternalFilesDir(null),
                "aicamera3rd-test-photo.jpg"
            ).canonicalFile
            val suppliedFile = File(suppliedPath).canonicalFile
            require(suppliedFile == expectedFile && suppliedFile.isFile) {
                "Rejected CI benchmark input path"
            }

            val mode = intent.getStringExtra(ImageProcessingService.EXTRA_MODE)
                ?: ColorEnhancer.Mode.AI.name
            val serviceIntent = Intent(this, ImageProcessingService::class.java).apply {
                putExtra(ImageProcessingService.EXTRA_INPUT, suppliedFile.absolutePath)
                putExtra(ImageProcessingService.EXTRA_MODE, mode)
                putExtra(ImageProcessingService.EXTRA_BENCHMARK, true)
            }
            ContextCompat.startForegroundService(this, serviceIntent)
            Log.i(TAG, "BENCHMARK_SERVICE_STARTED")
        } catch (t: Throwable) {
            Log.e(TAG, "BENCHMARK_START_FAILED", t)
        } finally {
            finish()
        }
    }

    companion object {
        private const val TAG = "AICameraBenchmark"
    }
}
