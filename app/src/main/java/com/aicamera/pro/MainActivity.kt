package com.aicamera.pro

import android.Manifest
import android.content.ContentValues
import android.content.BroadcastReceiver
import android.content.Intent
import android.content.IntentFilter
import android.content.ContentUris
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.Gravity
import android.widget.LinearLayout
import android.widget.ImageView
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

    private var capture: ImageCapture? = null
    @Volatile private var lensFacing = androidx.camera.core.CameraSelector.LENS_FACING_BACK
    @Volatile private var mode = ColorEnhancer.Mode.AI
    @Volatile private var maximEnhancer: MaximEnhancer? = null
    @Volatile private var faceProtector: FaceProtector? = null
    @Volatile private var denoiser: NafNetDenoiser? = null
    @Volatile private var sceneSegmenter: SceneSegmenter? = null
    @Volatile private var maximError: String? = null
    @Volatile private var denoiserError: String? = null
    private var busy = false
    private var progressReceiverRegistered = false
    private val processingProgressReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: android.content.Context?, intent: Intent?) {
            if (intent?.action != ImageProcessingService.ACTION_PROGRESS) return
            val title = intent.getStringExtra(ImageProcessingService.EXTRA_PROGRESS_TITLE) ?: "Processing"
            val percent = intent.getIntExtra(ImageProcessingService.EXTRA_PROGRESS_PERCENT, 0)
            val active = intent.getBooleanExtra(ImageProcessingService.EXTRA_PROGRESS_ACTIVE, false)
            val compact = when (title) {
                "Applying color enhancement", "AI color model running", "Enhancing vivid colors" -> "Enhancing"
                "Refining scene and face details" -> "Refining"
                "Loading offline AI models", "Loading portrait segmentation" -> "Loading"
                "Detecting portrait subject" -> "Detecting"
                "Saving to Gallery" -> "Saving"
                "Photo ready", "AI model ready" -> "Ready"
                else -> title
            }
            setStatus(if (active) "AI • $compact • $percent%" else if (percent == 100) "SAVED • 100%" else "PROCESSING FAILED")
        }
    }

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else toast("Camera permission required")
    }

    private val notificationPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        val selectedMode = mode
        setBusy(true, "Preparing photo for background processing…")
        processingExecutor.execute {
            val input = File(cacheDir, "import_${System.currentTimeMillis()}.img")
            try {
                contentResolver.openInputStream(uri)?.use { source ->
                    input.outputStream().use { target -> source.copyTo(target) }
                } ?: throw IOException("Could not read selected photo")
                startBackgroundProcessing(input, selectedMode)
            } catch (e: Exception) {
                input.delete()
                showFailure("Could not start photo processing: ${e.message ?: e.javaClass.simpleName}")
            }
        }
    }

    override fun onStart() {
        super.onStart()
        if (!progressReceiverRegistered) {
            ContextCompat.registerReceiver(
                this, processingProgressReceiver,
                IntentFilter(ImageProcessingService.ACTION_PROGRESS),
                ContextCompat.RECEIVER_NOT_EXPORTED
            )
            progressReceiverRegistered = true
        }
        val progress = getSharedPreferences(ImageProcessingService.PROGRESS_PREFS, MODE_PRIVATE)
        if (progress.getBoolean("active", false)) {
            val title = progress.getString("title", "Processing") ?: "Processing"
            val percent = progress.getInt("percent", 0)
            setStatus("AI • $title • $percent%")
        }
    }

    override fun onStop() {
        if (progressReceiverRegistered) {
            unregisterReceiver(processingProgressReceiver)
            progressReceiverRegistered = false
        }
        super.onStop()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Show a previously saved crash report before camera/model startup can fail again.
        CrashReporter.offerPending(this)
        cameraExecutor = Executors.newSingleThreadExecutor()
        processingExecutor = Executors.newSingleThreadExecutor()
        buildUi()

        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            permissions.launch(Manifest.permission.CAMERA)
        }
    }

    /**
     * Heavy AI runtimes are initialized only when a user requests image processing.
     * The launch path therefore does not touch TFLite, ML Kit or vendor GPU drivers.
     */
    private fun ensureAiModels() {
        // Stabilization path: load only MAXIM. NAFNet and the optional full-frame
        // scene/face models are deferred until their combined memory cost is measured.
        if (maximEnhancer == null) {
            maximEnhancer = try {
                MaximEnhancer(this)
            } catch (e: Exception) {
                maximError = e.message ?: "MAXIM initialization failed"
                throw IllegalStateException("MAXIM model could not be initialized: ${maximError}", e)
            }
        }
    }

    private fun enhance(bitmap: Bitmap, selectedMode: ColorEnhancer.Mode): Bitmap {
        if (selectedMode == ColorEnhancer.Mode.VIVID) {
            return ColorEnhancer.enhance(bitmap, ColorEnhancer.Mode.VIVID)
        }
        runOnUiThread { setStatus("AI • LOADING MAXIM…") }
        ensureAiModels()
        val maximModel = maximEnhancer
            ?: throw IllegalStateException(maximError ?: "MAXIM model is unavailable")

        // Cap the gallery AI path at 1024px on the long edge to prevent large
        // bitmap allocations and reduce CPU pressure on Snapdragon 865.
        val scale = minOf(1.0, 1024.0 / maxOf(bitmap.width, bitmap.height))
        val aiInput = if (scale < 1.0) {
            Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * scale).toInt().coerceAtLeast(1),
                (bitmap.height * scale).toInt().coerceAtLeast(1),
                true
            )
        } else bitmap
        try {
            runOnUiThread { setStatus("AI • ENHANCING COLORS…") }
            return maximModel.enhance(aiInput) { row, total ->
                if (row == total || row % 24 == 0) {
                    val percent = 30 + row * 65 / total.coerceAtLeast(1)
                    runOnUiThread { setStatus("AI • ENHANCING ${percent}%") }
                }
            }
        } finally {
            if (aiInput !== bitmap && !aiInput.isRecycled) aiInput.recycle()
        }
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
        topBar.addView(status, LinearLayout.LayoutParams(0, dp(32), 1f).apply {
            gravity = Gravity.CENTER_VERTICAL
            marginStart = dp(8)
        })
        val switchCameraButton = TextView(this).apply {
            text = "↻"
            textSize = 27f
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            contentDescription = "Switch front and back camera"
            background = rounded(0x66000000, 50)
            setOnClickListener {
                if (!busy) switchCamera()
            }
        }
        topBar.addView(switchCameraButton, LinearLayout.LayoutParams(dp(44), dp(44)).apply {
            marginStart = dp(10)
        })
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
            setStatus("AI • OFFLINE MODELS LOAD ON FIRST CAPTURE")
        }
        vividModeButton = modeChip("VIVID", false) {
            mode = ColorEnhancer.Mode.VIVID
            updateModeUi()
            setStatus("VIVID • COLOR ENHANCEMENT")
        }
        aiModeButton = modeChip("AI", true) {
            mode = ColorEnhancer.Mode.AI
            updateModeUi()
            setStatus("AI • OFFLINE MODELS LOAD ON FIRST CAPTURE")
        }
        vividModeButton = modeChip("VIVID", false) {
            mode = ColorEnhancer.Mode.VIVID
            updateModeUi()
            setStatus("VIVID • COLOR ENHANCEMENT")
        }
        modeRow.addView(aiModeButton, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginEnd = dp(5) })
        modeRow.addView(vividModeButton, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginStart = dp(5) })
        bottomPanel.addView(modeRow, LinearLayout.LayoutParams(-1, dp(44)).apply { bottomMargin = dp(12) })

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
        val recentButton = TextView(this).apply {
            text = "▣"
            textSize = 25f
            gravity = Gravity.CENTER
            setTextColor(Color.WHITE)
            contentDescription = "View captured photos"
            background = rounded(0x55333333, 50)
            setOnClickListener { if (!busy) openCapturedPhotos() }
        }
        captureRow.addView(recentButton, LinearLayout.LayoutParams(dp(54), dp(54)).apply { marginEnd = dp(22) })
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

        // Match the stock camera's tall central viewfinder without going edge-to-edge:
        // keep the top status bar and bottom shooting controls outside the preview.
        root.post {
            val availableWidth = root.width
            val topReserved = topBar.height
            val bottomReserved = bottomPanel.height
            val previewHeight = root.height - topReserved - bottomReserved - dp(16)
            if (availableWidth > 0 && previewHeight > 0) {
                preview.layoutParams = android.widget.FrameLayout.LayoutParams(
                    availableWidth,
                    previewHeight,
                    Gravity.TOP or Gravity.CENTER_HORIZONTAL
                ).apply {
                    topMargin = topReserved + dp(8)
                }
            }
        }

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
        styleModeButton(aiModeButton, mode == ColorEnhancer.Mode.AI)
        styleModeButton(vividModeButton, mode == ColorEnhancer.Mode.VIVID)
    }

    private fun styleModeButton(button: TextView, selected: Boolean) {
        button.setTextColor(if (selected) Color.BLACK else Color.WHITE)
        button.background = rounded(if (selected) Color.WHITE else 0x55333333, 24)
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
                val cameraPreview = Preview.Builder()
                    .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3)
                    .build().also {
                        it.setSurfaceProvider(preview.surfaceProvider)
                    }
                capture = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                    .setTargetAspectRatio(androidx.camera.core.AspectRatio.RATIO_4_3)
                    .build()
                provider.unbindAll()
                provider.bindToLifecycle(
                    this,
                    androidx.camera.core.CameraSelector.Builder()
                        .requireLensFacing(lensFacing)
                        .build(),
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


    private fun switchCamera() {
        lensFacing = if (lensFacing == androidx.camera.core.CameraSelector.LENS_FACING_BACK) {
            androidx.camera.core.CameraSelector.LENS_FACING_FRONT
        } else {
            androidx.camera.core.CameraSelector.LENS_FACING_BACK
        }
        setStatus(if (lensFacing == androidx.camera.core.CameraSelector.LENS_FACING_FRONT) {
            "FRONT CAMERA • 4:3 PREVIEW"
        } else {
            "BACK CAMERA • 4:3 PREVIEW"
        })
        startCamera()
    }

    private fun startBackgroundProcessing(input: File, selectedMode: ColorEnhancer.Mode) {
        try {
            val serviceIntent = Intent(this, ImageProcessingService::class.java).apply {
                putExtra(ImageProcessingService.EXTRA_INPUT, input.absolutePath)
                putExtra(ImageProcessingService.EXTRA_MODE, selectedMode.name)
            }
            androidx.core.content.ContextCompat.startForegroundService(this, serviceIntent)
            runOnUiThread {
                setBusy(false, "PROCESSING IN BACKGROUND • CHECK NOTIFICATION")
                toast("Photo processing started. You can minimize the app or lock the screen; it will save to Gallery when finished.")
            }
        } catch (e: Exception) {
            input.delete()
            showFailure("Could not start background processing: ${e.message ?: e.javaClass.simpleName}")
        }
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
                startBackgroundProcessing(output, selectedMode)
            }
        })
    }


    /**
     * Decode large phone photos safely. ImageDecoder supports modern formats (including HEIC)
     * on Android 9+, and all processing is capped at 2560px on the long edge to avoid
     * multi-hundred-megabyte bitmap allocations on mobile devices.
     */
    private fun decodeGalleryImage(uri: android.net.Uri): Bitmap {
        // 8MP-class processing cap balances detail with Snapdragon-class mobile performance.
        val maxEdge = 2048
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            val source = android.graphics.ImageDecoder.createSource(contentResolver, uri)
            return android.graphics.ImageDecoder.decodeBitmap(source) { decoder, info, _ ->
                val width = info.size.width
                val height = info.size.height
                val scale = minOf(1.0, maxEdge.toDouble() / maxOf(width, height))
                decoder.setTargetSize(
                    (width * scale).toInt().coerceAtLeast(1),
                    (height * scale).toInt().coerceAtLeast(1)
                )
                decoder.allocator = android.graphics.ImageDecoder.ALLOCATOR_SOFTWARE
            }
        }

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
            ?: throw IOException("Could not open selected photo")
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("Unsupported or damaged image")
        val options = BitmapFactory.Options().apply {
            inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight, maxEdge)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, options) }
            ?: throw IOException("Could not decode selected photo")
    }

    private fun decodeCapturedImage(file: File): Bitmap {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) throw IOException("Captured image is invalid")
        val options = BitmapFactory.Options().apply {
            inSampleSize = calculateSampleSize(bounds.outWidth, bounds.outHeight, 2048)
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return BitmapFactory.decodeFile(file.absolutePath, options)
            ?: throw IOException("Captured image could not be decoded")
    }

    private fun calculateSampleSize(width: Int, height: Int, maxEdge: Int): Int {
        var sample = 1
        while (maxOf(width / sample, height / sample) > maxEdge) sample *= 2
        return sample
    }


    /** Gallery imports require explicit Save; Cancel discards the enhanced result. */
    private fun processGalleryForPreview(bitmap: Bitmap, selectedMode: ColorEnhancer.Mode) {
        var enhanced: Bitmap? = null
        try {
            runOnUiThread { setStatus("Processing gallery photo • ${selectedMode.name}…") }
            enhanced = enhance(bitmap, selectedMode)
            if (enhanced !== bitmap && !bitmap.isRecycled) bitmap.recycle()
            val result = enhanced ?: throw IllegalStateException("Enhancement returned no image")
            enhanced = null
            runOnUiThread { showGalleryResult(result, selectedMode) }
        } catch (e: Exception) {
            showFailure("Photo processing failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            if (enhanced != null && enhanced !== bitmap && !enhanced.isRecycled) enhanced.recycle()
            if (!bitmap.isRecycled) bitmap.recycle()
        }
    }

    private fun showGalleryResult(result: Bitmap, selectedMode: ColorEnhancer.Mode) {
        val image = ImageView(this).apply {
            setImageBitmap(result)
            scaleType = ImageView.ScaleType.FIT_CENTER
            setBackgroundColor(Color.BLACK)
            adjustViewBounds = true
        }
        val content = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(12), dp(12), dp(12), dp(4))
            setBackgroundColor(Color.BLACK)
            addView(image, LinearLayout.LayoutParams(-1, dp(430)))
        }
        val dialog = androidx.appcompat.app.AlertDialog.Builder(this)
            .setTitle("Enhanced photo • ${selectedMode.name}")
            .setView(content)
            .setPositiveButton("SAVE", null)
            .setNegativeButton("CANCEL", null)
            .create()
        dialog.setOnShowListener {
            dialog.getButton(androidx.appcompat.app.AlertDialog.BUTTON_POSITIVE).setOnClickListener {
                try {
                    saveBitmap(result)
                    if (!result.isRecycled) result.recycle()
                    dialog.dismiss()
                    setBusy(false, "SAVED • Pictures/AICamera3rd")
                    toast("Enhanced photo saved to Gallery")
                } catch (e: Exception) {
                    toast("Save failed: ${e.message ?: e.javaClass.simpleName}")
                }
            }
            dialog.getButton(androidx.appcompat.app.AlertDialog.BUTTON_NEGATIVE).setOnClickListener {
                if (!result.isRecycled) result.recycle()
                dialog.dismiss()
                setBusy(false, "READY • Camera")
                toast("Gallery edit cancelled")
            }
        }
        dialog.setOnCancelListener {
            if (!result.isRecycled) result.recycle()
            setBusy(false, "READY • Camera")
        }
        dialog.show()
        dialog.window?.setLayout((resources.displayMetrics.widthPixels * 0.94f).toInt(), android.view.WindowManager.LayoutParams.WRAP_CONTENT)
        setStatus("REVIEW • SAVE OR CANCEL")
    }

    /** Open the newest photo saved by this app in the phone's image viewer. */
    private fun openCapturedPhotos() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            toast("Captured photos are saved in Pictures/AICamera3rd")
            return
        }
        val projection = arrayOf(MediaStore.Images.Media._ID)
        val selection = "${MediaStore.Images.Media.RELATIVE_PATH} = ?"
        val selectionArgs = arrayOf("Pictures/AICamera3rd/")
        val sortOrder = "${MediaStore.Images.Media.DATE_ADDED} DESC"
        val uri = contentResolver.query(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, projection, selection, selectionArgs, sortOrder)?.use { cursor ->
            if (cursor.moveToFirst()) ContentUris.withAppendedId(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID))
            ) else null
        }
        if (uri == null) { toast("No captured photos yet"); return }
        try {
            startActivity(Intent(Intent.ACTION_VIEW).apply {
                setDataAndType(uri, "image/*")
                addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
            })
        } catch (_: Exception) { toast("No photo viewer is available") }
    }

    private fun processAndSave(bitmap: Bitmap, selectedMode: ColorEnhancer.Mode) {
        var enhanced: Bitmap? = null
        try {
            runOnUiThread { setStatus("Processing photo • ${selectedMode.name}…") }
            enhanced = enhance(bitmap, selectedMode)
            runOnUiThread { setStatus("Saving enhanced photo to Gallery…") }
            saveBitmap(enhanced)
            runOnUiThread {
                setBusy(false, "SAVED • Pictures/AICamera3rd")
                toast("${selectedMode.name} photo saved to Gallery")
            }
        } catch (e: Exception) {
            showFailure("Enhancement/save failed: ${e.message ?: e.javaClass.simpleName}")
        } finally {
            if (enhanced != null && enhanced !== bitmap && !enhanced.isRecycled) enhanced.recycle()
            if (!bitmap.isRecycled) bitmap.recycle()
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
        maximEnhancer?.close()
        denoiser?.close()
        sceneSegmenter?.close()
        faceProtector?.close()
        super.onDestroy()
    }
}
