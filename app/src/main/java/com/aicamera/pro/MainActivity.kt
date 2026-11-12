package com.aicamera.pro

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private lateinit var preview: PreviewView
    private lateinit var status: TextView
    private lateinit var aiModeButton: TextView
    private lateinit var vividModeButton: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var processingExecutor: ExecutorService
    private lateinit var modelExecutor: ExecutorService

    private var capture: ImageCapture? = null
    @Volatile private var mode = ColorEnhancer.Mode.AI
    @Volatile private var maximEnhancer: MaximEnhancer? = null
    @Volatile private var faceProtector: FaceProtector? = null
    @Volatile private var denoiser: NafNetDenoiser? = null
    @Volatile private var sceneSegmenter: SceneSegmenter? = null
    @Volatile private var maximError: String? = null
    @Volatile private var denoiserError: String? = null
    private var busy = false

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else toast("Camera permission required")
    }

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        val selectedMode = mode
        setBusy(true, "Opening selected photo…")
        processingExecutor.execute {
            try {
                val bitmap = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
                    ?: throw IOException("Could not read selected photo")
                processAndPreview(bitmap, selectedMode)
            } catch (e: Exception) {
                showFailure("Photo processing failed: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        cameraExecutor = Executors.newSingleThreadExecutor()
        processingExecutor = Executors.newSingleThreadExecutor()
        modelExecutor = Executors.newSingleThreadExecutor()
        buildUi()
        loadModels()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permissions.launch(Manifest.permission.CAMERA)
        }
    }

    private fun loadModels() {
        setStatus("Preparing offline AI models…")
        modelExecutor.execute {
            // Load each model independently: one optional model failure must not disable the others.
            try {
                maximEnhancer = MaximEnhancer(this)
            } catch (e: Exception) {
                maximError = e.message ?: "MAXIM initialization failed"
            }
            try {
                denoiser = NafNetDenoiser(this)
            } catch (e: Exception) {
                denoiserError = e.message ?: "NAFNet initialization failed"
            }
            try {
                sceneSegmenter = SceneSegmenter(this)
            } catch (_: Exception) {
                sceneSegmenter = null
            }
            try {
                faceProtector = FaceProtector()
            } catch (_: Exception) {
                faceProtector = null
            }

            runOnUiThread {
                val ready = maximEnhancer != null && denoiser != null
                if (ready) {
                    setStatus("AI READY • MAXIM + NAFNET • OFFLINE")
                } else {
                    val reason = listOfNotNull(denoiserError, maximError).joinToString(" • ")
                    setStatus("AI MODEL INCOMPLETE • Vivid mode is available")
                    if (reason.isNotBlank()) toast(reason)
                }
            }
        }
    }

    private fun enhance(bitmap: Bitmap, selectedMode: ColorEnhancer.Mode): Bitmap {
        if (selectedMode == ColorEnhancer.Mode.VIVID) {
            return ColorEnhancer.enhance(bitmap, ColorEnhancer.Mode.VIVID)
        }

        val denoiserModel = denoiser
            ?: throw IllegalStateException(denoiserError ?: "NAFNet is still loading. Please wait.")
        val maximModel = maximEnhancer
            ?: throw IllegalStateException(maximError ?: "MAXIM is still loading. Please wait.")

        val denoised = denoiserModel.denoise(bitmap)
        val maxim = maximModel.enhance(denoised)
        val sceneAware = sceneSegmenter?.let { it.apply(denoised, maxim) } ?: maxim
        return faceProtector?.let { it.protect(bitmap, sceneAware) } ?: sceneAware
    }

    private fun buildUi() {
        window.statusBarColor = Color.BLACK
        window.navigationBarColor = Color.BLACK

        val root = android.widget.FrameLayout(this).apply {
            setBackgroundColor(Color.BLACK)
        }

        preview = PreviewView(this).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
            setBackgroundColor(Color.BLACK)
        }
        root.addView(preview, android.widget.FrameLayout.LayoutParams(-1, -1))

        // Minimal camera-style top bar: brand at left, compact processing status at right.
        val topBar = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(dp(20), dp(10), dp(20), dp(10))
            background = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(0x99000000.toInt(), 0x00000000)
            )
        }
        val brand = TextView(this).apply {
            text = "AI CAMERA"
            setTextColor(Color.WHITE)
            textSize = 14f
            letterSpacing = 0.12f
            setTypeface(null, android.graphics.Typeface.BOLD)
        }
        topBar.addView(brand, LinearLayout.LayoutParams(0, dp(42), 1f))
        status = TextView(this).apply {
            text = "AI • LOADING"
            setTextColor(Color.WHITE)
            textSize = 10f
            gravity = Gravity.CENTER
            setPadding(dp(12), dp(7), dp(12), dp(7))
            background = rounded(0x66000000, 24)
            maxLines = 1
        }
        topBar.addView(status, LinearLayout.LayoutParams(-2, dp(32)))
        root.addView(topBar, android.widget.FrameLayout.LayoutParams(-1, dp(76), Gravity.TOP))

        // Camera controls sit over a subtle bottom scrim; the viewfinder stays dominant.
        val bottomPanel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setPadding(dp(20), dp(28), dp(20), dp(26))
            background = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(0x00000000, 0xCC000000.toInt(), 0xF2000000.toInt())
            )
        }

        val modeRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        aiModeButton = modeChip("AI", true) {
            mode = ColorEnhancer.Mode.AI
            updateModeUi()
            setStatus(if (maximEnhancer != null && denoiser != null) "AI • READY" else "AI • LOADING MODELS")
        }
        vividModeButton = modeChip("VIVID", false) {
            mode = ColorEnhancer.Mode.VIVID
            updateModeUi()
            setStatus("VIVID • COLOR ENHANCEMENT")
        }
        modeRow.addView(aiModeButton, LinearLayout.LayoutParams(dp(104), dp(40)).apply { marginEnd = dp(8) })
        modeRow.addView(vividModeButton, LinearLayout.LayoutParams(dp(104), dp(40)).apply { marginStart = dp(8) })
        bottomPanel.addView(modeRow, LinearLayout.LayoutParams(-1, dp(48)).apply {
            bottomMargin = dp(18)
        })

        val captureRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        val galleryButton = TextView(this).apply {
            text = "▧"
            textSize = 27f
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            background = rounded(0x55333333, 50)
            setOnClickListener { if (!busy) picker.launch("image/*") }
        }
        captureRow.addView(galleryButton, LinearLayout.LayoutParams(dp(54), dp(54)).apply {
            marginStart = dp(22)
        })
        val shutter = android.widget.FrameLayout(this).apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.WHITE)
                setStroke(dp(4), 0x99FFFFFF.toInt())
            }
            isClickable = true
            isFocusable = true
            setOnClickListener { takePhoto() }
        }
        val shutterInner = android.view.View(this).apply {
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(Color.WHITE)
                setStroke(dp(2), Color.BLACK)
            }
        }
        shutter.addView(shutterInner, android.widget.FrameLayout.LayoutParams(dp(46), dp(46), Gravity.CENTER))
        captureRow.addView(shutter, LinearLayout.LayoutParams(dp(68), dp(68)).apply {
            weight = 1f
            gravity = Gravity.CENTER
        })
        val spacer = TextView(this).apply { text = "" }
        captureRow.addView(spacer, LinearLayout.LayoutParams(dp(54), dp(54)).apply {
            marginEnd = dp(22)
        })
        bottomPanel.addView(captureRow, LinearLayout.LayoutParams(-1, dp(76)))

        val hint = TextView(this).apply {
            text = "Tap to capture  •  AI enhancement"
            setTextColor(0xBFFFFFFF.toInt())
            textSize = 11f
            gravity = Gravity.CENTER
        }
        bottomPanel.addView(hint, LinearLayout.LayoutParams(-1, dp(24)).apply {
            topMargin = dp(6)
        })

        root.addView(bottomPanel, android.widget.FrameLayout.LayoutParams(-1, -2, Gravity.BOTTOM))
        setContentView(root)
        updateModeUi()
    }

    private fun modeChip(label: String, selected: Boolean, action: () -> Unit): TextView {
        return TextView(this).apply {
            text = label
            gravity = Gravity.CENTER
            textSize = 12f
            letterSpacing = 0.08f
            setTypeface(null, android.graphics.Typeface.BOLD)
            setOnClickListener { action() }
            tag = selected
        }
    }

    private fun updateModeUi() {
        if (!::aiModeButton.isInitialized || !::vividModeButton.isInitialized) return
        val aiSelected = mode == ColorEnhancer.Mode.AI
        aiModeButton.apply {
            setTextColor(if (aiSelected) Color.BLACK else Color.WHITE)
            background = rounded(if (aiSelected) Color.WHITE else 0x55333333, 24)
        }
        vividModeButton.apply {
            setTextColor(if (!aiSelected) Color.BLACK else Color.WHITE)
            background = rounded(if (!aiSelected) Color.WHITE else 0x55333333, 24)
        }
    }

    private fun rounded(color: Int, radiusDp: Int) = GradientDrawable().apply {
        setColor(color)
        cornerRadius = dp(radiusDp).toFloat()
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                val cameraPreview = Preview.Builder().build().also {
                    it.setSurfaceProvider(preview.surfaceProvider)
                }
                capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3)
                    .build()
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    androidx.camera.core.CameraSelector.DEFAULT_BACK_CAMERA,
                    cameraPreview,
                    capture
                )
                setStatus("CAMERA READY • Choose AI or Vivid")
            } catch (e: Exception) {
                setStatus("CAMERA ERROR • ${e.localizedMessage ?: "Unable to start"}")
                toast("Camera start failed")
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun takePhoto() {
        if (busy) return
        val imageCapture = capture
        if (imageCapture == null) {
            toast("Camera is not ready yet")
            return
        }

        val selectedMode = mode
        val output = File(cacheDir, "capture_${System.currentTimeMillis()}.jpg")
        setBusy(true, "Capturing photo…")
        val options = ImageCapture.OutputFileOptions.Builder(output).build()

        imageCapture.takePicture(options, cameraExecutor, object : ImageCapture.OnImageSavedCallback {
            override fun onError(exc: ImageCaptureException) {
                output.delete()
                showFailure("Capture failed: ${exc.message ?: "Camera error"}")
            }

            override fun onImageSaved(result: ImageCapture.OutputFileResults) {
                processingExecutor.execute {
                    try {
                        val bitmap = BitmapFactory.decodeFile(output.absolutePath)
                            ?: throw IOException("Captured image could not be decoded")
                        processAndPreview(bitmap, selectedMode)
                    } catch (e: Exception) {
                        showFailure("Enhancement failed: ${e.message ?: e.javaClass.simpleName}")
                    } finally {
                        output.delete()
                    }
                }
            }
        })
    }

    private fun processAndPreview(bitmap: Bitmap, selectedMode: ColorEnhancer.Mode) {
        try {
            runOnUiThread { setStatus("Processing photo • ${selectedMode.name}…") }
            val enhanced = enhance(bitmap, selectedMode)
            runOnUiThread {
                setBusy(false, "Enhancement complete • Review and save")
                showSavePreview(enhanced)
            }
        } catch (e: Exception) {
            showFailure("Enhancement failed: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private fun showSavePreview(bitmap: Bitmap) {
        val image = ImageView(this).apply {
            setImageBitmap(bitmap)
            adjustViewBounds = true
            scaleType = ImageView.ScaleType.FIT_CENTER
            maxHeight = (resources.displayMetrics.heightPixels * 0.55f).toInt()
            setBackgroundColor(Color.BLACK)
        }
        val scroll = ScrollView(this).apply {
            addView(image, ViewGroup.LayoutParams(-1, ViewGroup.LayoutParams.WRAP_CONTENT))
        }
        val dialog = AlertDialog.Builder(this)
            .setTitle("Enhanced photo")
            .setView(scroll)
            .setPositiveButton("SAVE TO GALLERY", null)
            .setNegativeButton("DISCARD", null)
            .create()

        dialog.setOnShowListener {
            dialog.getButton(AlertDialog.BUTTON_POSITIVE).setOnClickListener { savePreview(bitmap, dialog) }
        }
        dialog.show()
        dialog.window?.setBackgroundDrawableResource(android.R.color.background_dark)
        dialog.getButton(AlertDialog.BUTTON_POSITIVE)?.setTextColor(Color.rgb(30, 150, 105))
    }

    private fun savePreview(bitmap: Bitmap, dialog: AlertDialog) {
        dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = false
        setBusy(true, "Saving photo to Pictures/AICamera3rd…")
        processingExecutor.execute {
            try {
                val uri = saveBitmap(bitmap)
                runOnUiThread {
                    dialog.dismiss()
                    setBusy(false, "SAVED • Pictures/AICamera3rd")
                    toast("Photo saved to Gallery")
                }
            } catch (e: Exception) {
                runOnUiThread {
                    dialog.getButton(AlertDialog.BUTTON_POSITIVE).isEnabled = true
                    setBusy(false, "Save failed • Please try again")
                    toast("Save failed: ${e.message ?: e.javaClass.simpleName}")
                }
            }
        }
    }

    @Throws(IOException::class)
    private fun saveBitmap(bitmap: Bitmap): android.net.Uri {
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "AICamera3rd_$stamp.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/AICamera3rd")
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }

        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
            ?: throw IOException("Android could not create the gallery file")

        try {
            val stream = contentResolver.openOutputStream(uri)
                ?: throw IOException("Could not open the gallery output stream")
            stream.use {
                if (!bitmap.compress(Bitmap.CompressFormat.JPEG, 96, it)) {
                    throw IOException("JPEG encoding failed")
                }
                it.flush()
            }
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                val publish = ContentValues().apply { put(MediaStore.Images.Media.IS_PENDING, 0) }
                contentResolver.update(uri, publish, null, null)
            }
            return uri
        } catch (e: Exception) {
            contentResolver.delete(uri, null, null)
            throw e
        }
    }

    private fun setBusy(value: Boolean, message: String) {
        busy = value
        setStatus(message)
    }

    private fun setStatus(message: String) {
        if (::status.isInitialized) status.text = message
    }

    private fun showFailure(message: String) {
        runOnUiThread {
            setBusy(false, message)
            toast(message)
        }
    }

    private fun dp(value: Int): Int = (value * resources.displayMetrics.density).toInt()
    private fun toast(message: String) = Toast.makeText(this, message, Toast.LENGTH_LONG).show()

    override fun onDestroy() {
        cameraExecutor.shutdownNow()
        processingExecutor.shutdownNow()
        modelExecutor.shutdownNow()
        maximEnhancer?.close()
        denoiser?.close()
        sceneSegmenter?.close()
        faceProtector?.close()
        super.onDestroy()
    }
}
