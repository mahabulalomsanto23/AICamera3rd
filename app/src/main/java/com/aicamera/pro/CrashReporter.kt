package com.aicamera.pro

import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.appcompat.app.AlertDialog
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/** Captures uncaught Java/Kotlin exceptions from the entire app process. */
object CrashReporter {
    private const val FILE_NAME = "last-crash-report.txt"
    @Volatile private var pendingReport: String? = null

    fun install(context: Context) {
        val appContext = context.applicationContext
        val file = File(appContext.filesDir, FILE_NAME)
        pendingReport = try {
            if (file.exists()) file.readText().takeIf { it.isNotBlank() } else null
        } catch (_: Exception) { null }

        val previousHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val trace = StringWriter()
                throwable.printStackTrace(PrintWriter(trace))
                val report = buildString {
                    appendLine("AI Camera Pro crash report")
                    appendLine("Time: " + SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.US).format(Date()))
                    appendLine("Device: " + Build.MANUFACTURER + " " + Build.MODEL)
                    appendLine("Android: " + Build.VERSION.RELEASE + " (SDK " + Build.VERSION.SDK_INT + ")")
                    appendLine("Thread: " + thread.name)
                    appendLine()
                    append(trace.toString())
                }
                val temp = File(appContext.filesDir, "$FILE_NAME.tmp")
                temp.writeText(report)
                val target = file
                if (target.exists()) target.delete()
                if (!temp.renameTo(target)) target.writeText(report)
            } catch (_: Throwable) { /* Never block Android's crash handler. */ }

            if (previousHandler != null) previousHandler.uncaughtException(thread, throwable)
            else android.os.Process.killProcess(android.os.Process.myPid())
        }
    }

    fun offerPending(context: Context) {
        val report = pendingReport ?: return
        pendingReport = null
        AlertDialog.Builder(context)
            .setTitle("আগেরবার অ্যাপ বন্ধ হয়ে গিয়েছিল")
            .setMessage("Crash report পাওয়া গেছে। SHARE LOG চাপলে রিপোর্টটি পাঠাতে পারবেন।")
            .setPositiveButton("SHARE LOG") { _, _ ->
                val send = Intent(Intent.ACTION_SEND).apply {
                    type = "text/plain"
                    putExtra(Intent.EXTRA_SUBJECT, "AI Camera Pro crash report")
                    putExtra(Intent.EXTRA_TEXT, report)
                }
                context.startActivity(Intent.createChooser(send, "Share crash report"))
            }
            .setNegativeButton("LATER", null)
            .show()
    }
}
