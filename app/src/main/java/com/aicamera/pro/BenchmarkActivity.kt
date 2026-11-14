package com.aicamera.pro

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.core.content.ContextCompat

/**
 * CI-only entry point for exercising the real offline AI service.
 * Launch with MODE=AI and INPUT=benchmark://generated.
 */
class BenchmarkActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            val input = intent.getStringExtra(ImageProcessingService.EXTRA_INPUT)
                ?: "benchmark://generated"
            val mode = intent.getStringExtra(ImageProcessingService.EXTRA_MODE) ?: ColorEnhancer.Mode.AI.name
            val serviceIntent = Intent(this, ImageProcessingService::class.java).apply {
                putExtra(ImageProcessingService.EXTRA_INPUT, input)
                putExtra(ImageProcessingService.EXTRA_MODE, mode)
                putExtra(ImageProcessingService.EXTRA_BENCHMARK, true)
            }
            ContextCompat.startForegroundService(this, serviceIntent)
            Log.i("AICameraBenchmark", "BENCHMARK_START_OK input=$input mode=$mode")
        } catch (t: Throwable) {
            Log.e("AICameraBenchmark", "BENCHMARK_START_FAILED ${t.javaClass.simpleName}: ${t.message}", t)
            throw t
        } finally {
            finish()
        }
    }
}
