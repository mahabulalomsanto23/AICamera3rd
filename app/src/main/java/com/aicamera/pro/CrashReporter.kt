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

/**
 * Stores fatal app crashes locally so the user can share the report without a computer,
 * ADB, or developer tools. The report is offered on the next successful app launch.
 */
object CrashReporter {
    private const val FILE_NAME = "last-crash-report.txt"

    fun install(context: Context): String? {
        val file = File(context.filesDir, FILE_NAME)
        val previous = try {
            if (file.exists()) file.readText().takeIf { it.isNotBlank() } else null
        } catch (_: Exception) { null }

        val defaultHandler = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
            try {
                val writer = StringWriter()
                throwable.printStackTrace(PrintWriter(writer))
                val report = buildString {
                    appendLine("AI Camera Pro crash report")
                    appendLine("Time: " + SimpleDateFormat("yyyy-MM-dd HH:mm:ss Z", Locale.US).format(Date()))
                    val appInfo = context.packageManager.getPackageInfo(context.packageName, 0)
                    appendLine("App: " + appInfo.versionName + " (" + appInfo.versionCode + ")")
                    appendLine("Device: " + Build.MANUFACTURER + " " + Build.MODEL)
                    appendLine("Android: " + Build.VERSION.RELEASE + " (SDK " + Build.VERSION.SDK_INT + ")")
                    appendLine("Thread: " + thread.name)
                    appendLine()
                    append(writer.toString())
                }
                file.writeText(report)
            } catch (_: Exception) { }
            if (defaultHandler != null) defaultHandler.uncaughtException(thread, throwable)
            else android.os.Process.killProcess(android.os.Process.myPid())
        }
        return previous
    }

    fun offerShare(context: Context, report: String) {
        AlertDialog.Builder(context)
            .setTitle("আগেরবার অ্যাপ বন্ধ হয়ে গিয়েছিল")
            .setMessage("Crash report পাওয়া গেছে। এটি Share করলে আমি সমস্যাটি শনাক্ত করে ঠিক করতে পারব।")
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
