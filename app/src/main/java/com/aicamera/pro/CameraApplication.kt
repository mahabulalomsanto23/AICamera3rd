package com.aicamera.pro

import android.app.Application

class CameraApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        CrashReporter.install(this)
    }
}
