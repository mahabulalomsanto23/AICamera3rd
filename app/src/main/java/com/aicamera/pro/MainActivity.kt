package com.aicamera.pro

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var cameraExecutor: ExecutorService
    private var capture: ImageCapture? = null
    private var mode = ColorEnhancer.Mode.AI
    private val permissions = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else toast("Camera permission required")
    }
    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        uri ?: return@registerForActivityResult
        contentResolver.openInputStream(uri)?.use { stream ->
            val bitmap = BitmapFactory.decodeStream(stream) ?: return@use
            Thread {
                val result = ColorEnhancer.enhance(bitmap, mode)
                runOnUiThread { saveBitmap(result); status.text = "Enhanced photo saved" }
            }.start()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        buildUi()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) startCamera()
        else permissions.launch(Manifest.permission.CAMERA)
    }

    private fun buildUi() {
        val root = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setBackgroundColor(0xFF000000.toInt()) }
        preview = PreviewView(this).apply { scaleType = PreviewView.ScaleType.FILL_CENTER; implementationMode = PreviewView.ImplementationMode.COMPATIBLE }
        // Match the native camera framing: full device width, 4:3 image viewport.
        // The preview is NOT stretched to fill the remaining screen height.
        preview.post {
            val width = root.width.takeIf { it > 0 } ?: resources.displayMetrics.widthPixels
            preview.layoutParams = LinearLayout.LayoutParams(width, width * 3 / 4)
        }
        root.addView(preview, LinearLayout.LayoutParams(-1, -2))
        status = TextView(this).apply { text = "AI MODE • DAYLIGHT COLOR"; setTextColor(-1); gravity = Gravity.CENTER; textSize = 12f; setPadding(8,10,8,6) }
        root.addView(status)
        val modes = LinearLayout(this).apply { gravity = Gravity.CENTER; setPadding(8,4,8,8) }
        modes.addView(button("AI MODE") { mode = ColorEnhancer.Mode.AI; status.text = "AI MODE • OFFLINE COLOR PIPELINE" })
        modes.addView(button("VIVID") { mode = ColorEnhancer.Mode.VIVID; status.text = "VIVID MODE • RICH COLOR" })
        root.addView(modes)
        val controls = LinearLayout(this).apply { gravity = Gravity.CENTER; setPadding(8,4,8,22) }
        controls.addView(button("GALLERY") { picker.launch("image/*") })
        controls.addView(button("●") { takePhoto() })
        root.addView(controls)
        setContentView(root)
    }

    private fun button(label: String, action: () -> Unit) = Button(this).apply {
        text = label; isAllCaps = false; textSize = 13f; setOnClickListener { action() }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()
            val p = Preview.Builder().build().also { it.setSurfaceProvider(preview.surfaceProvider) }
            capture = ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3).build()
            try { provider.unbindAll(); provider.bindToLifecycle(this, androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA, p, capture) }
            catch (e: Exception) { toast("Camera start failed: ${e.localizedMessage}") }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        val output = File(getExternalFilesDir(null), "capture_${System.currentTimeMillis()}.jpg")
        val options = ImageCapture.OutputFileOptions.Builder(output).build()
        capture?.takePicture(options, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onError(exc: ImageCaptureException) { runOnUiThread { toast("Capture failed") } }
            override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                val bitmap = BitmapFactory.decodeFile(output.absolutePath)
                if (bitmap != null) {
                    val enhanced = ColorEnhancer.enhance(bitmap, mode)
                    runOnUiThread { saveBitmap(enhanced); toast("Photo saved") }
                }
                output.delete()
            }
        })
    }

    private fun saveBitmap(bitmap: android.graphics.Bitmap) {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "AICamera3rd_$stamp.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AICamera3rd")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
        contentResolver.openOutputStream(uri)?.use { bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 96, it) }
    }

    private fun toast(s: String) = Toast.makeText(this, s, Toast.LENGTH_SHORT).show()
    override fun onDestroy() { super.onDestroy(); cameraExecutor.shutdown() }
}
