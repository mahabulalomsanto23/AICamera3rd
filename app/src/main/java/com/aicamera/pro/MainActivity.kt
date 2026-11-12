package com.aicamera.pro

import android.Manifest
import android.content.ContentValues
import android.content.Intent
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
    private lateinit var portraitModeButton: TextView
    private lateinit var lowStrengthButton: TextView
    private lateinit var mediumStrengthButton: TextView
    private lateinit var highStrengthButton: TextView
    private lateinit var portraitStrengthRow: LinearLayout
    @Volatile private var portraitStrength = 0.72f
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var processingExecutor: ExecutorService
    private lateinit var modelExecutor: ExecutorService

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

    private val permissions = registerForActivityResult(ActivityResultContracts.RequestPermission()) {
        if (it) startCamera() else toast("Camera permission required")
    }

    private val picker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri == null) return@registerForActivityResult
        val selectedMode = mode
        setBusy(true, "Opening selected photo…")
        processingExecutor.execute {
            try {
                val bitmap = decodeGalleryImage(uri)
                processGalleryForPreview(bitmap, selectedMode)
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
        if (selectedMode == ColorEnhancer.Mode.PORTRAIT) {
            val segmenter = sceneSegmenter
                ?: throw IllegalStateException("Portrait model is still loading or unavailable. Please wait.")
            val personMask = segmenter.personMask(bitmap)
            return PortraitProcessor.apply(bitmap, personMask, portraitStrength)
        }

        val denoiserModel = denoiser
            ?: throw IllegalStateException(denoiserError ?: "NAFNet is still loading. Please wait.")
        val maximModel = maximEnhancer
            ?: throw IllegalStateException(maximError ?: "MAXIM is still loading. Please wait.")

        runOnUiThread { setStatus("AI • DENOISING DETAILS…") }
        val denoised = denoiserModel.denoise(bitmap) { done, total ->
            if (done == 1 || done == total || done % 4 == 0) {
                runOnUiThread { setStatus("AI • DENOISING $done/$total") }
            }
        }
        try {
            runOnUiThread { setStatus("AI • ENHANCING COLORS…") }
            val maxim = maximModel.enhance(denoised)
            try {
                runOnUiThread { setStatus("AI • SCENE & FACE REFINEMENT…") }
                val sceneAware = sceneSegmenter?.let { it.apply(denoised, maxim) } ?: maxim
                val finalBitmap = faceProtector?.let { it.protect(bitmap, sceneAware) } ?: sceneAware
                if (sceneAware !== finalBitmap && sceneAware !== maxim && !sceneAware.isRecycled) sceneAware.recycle()
                if (maxim !== finalBitmap && !maxim.isRecycled) maxim.recycle()
                return finalBitmap
            } catch (e: Exception) {
                if (!maxim.isRecycled) maxim.recycle()
                throw e
            }
        } finally {
            if (!denoised.isRecycled) denoised.recycle()
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
        root.post {
            val frameWidth = root.width
            if (frameWidth > 0) {
                preview.layoutParams = android.widget.FrameLayout.LayoutParams(frameWidth, (frameWidth * 3f / 4f).toInt(), Gravity.CENTER)
            }
        }

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
            setStatus(if (maximEnhancer != null && denoiser != null) "AI • READY" else "AI • LOADING MODELS")
        }
        vividModeButton = modeChip("VIVID", false) {
            mode = ColorEnhancer.Mode.VIVID
            updateModeUi()
            setStatus("VIVID • COLOR ENHANCEMENT")
        }
        portraitModeButton = modeChip("PORTRAIT", false) {
            mode = ColorEnhancer.Mode.PORTRAIT
            updateModeUi()
            setStatus(if (sceneSegmenter != null) "PORTRAIT • NATURAL DEPTH" else "PORTRAIT • LOADING MODEL")
        }
        modeRow.addView(aiModeButton, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginEnd = dp(4) })
        modeRow.addView(vividModeButton, LinearLayout.LayoutParams(0, dp(40), 1f).apply {
            marginStart = dp(4); marginEnd = dp(4)
        })
        modeRow.addView(portraitModeButton, LinearLayout.LayoutParams(0, dp(40), 1f).apply { marginStart = dp(4) })
        bottomPanel.addView(modeRow, LinearLayout.LayoutParams(-1, dp(44)).apply { bottomMargin = dp(4) })

        val strengthRow = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
        }
        lowStrengthButton = modeChip("LOW", false) {
            portraitStrength = 0.55f
            updatePortraitStrengthUi()
        }
        mediumStrengthButton = modeChip("MEDIUM", true) {
            portraitStrength = 0.72f
            updatePortraitStrengthUi()
        }
        highStrengthButton = modeChip("HIGH", false) {
            portraitStrength = 0.90f
            updatePortraitStrengthUi()
        }
        strengthRow.addView(lowStrengthButton, LinearLayout.LayoutParams(dp(76), dp(30)).apply { marginEnd = dp(6) })
        strengthRow.addView(mediumStrengthButton, LinearLayout.LayoutParams(dp(88), dp(30)).apply { marginStart = dp(3); marginEnd = dp(3) })
        strengthRow.addView(highStrengthButton, LinearLayout.LayoutParams(dp(76), dp(30)).apply { marginStart = dp(6) })
        bottomPanel.addView(strengthRow, LinearLayout.LayoutParams(-1, dp(34)).apply { bottomMargin = dp(10) })
        portraitStrengthRow = strengthRow

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
        if (!::aiModeButton.isInitialized || !::vividModeButton.isInitialized ||
            !::portraitModeButton.isInitialized) return
        styleModeButton(aiModeButton, mode == ColorEnhancer.Mode.AI)
        styleModeButton(vividModeButton, mode == ColorEnhancer.Mode.VIVID)
        styleModeButton(portraitModeButton, mode == ColorEnhancer.Mode.PORTRAIT)
        if (::portraitStrengthRow.isInitialized) {
            portraitStrengthRow.visibility = if (mode == ColorEnhancer.Mode.PORTRAIT) android.view.View.VISIBLE else android.view.View.GONE
        }
        updatePortraitStrengthUi()
    }

    private fun styleModeButton(button: TextView, selected: Boolean) {
        button.setTextColor(if (selected) Color.BLACK else Color.WHITE)
        button.background = rounded(if (selected) Color.WHITE else 0x55333333, 24)
    }

    private fun updatePortraitStrengthUi() {
        if (!::lowStrengthButton.isInitialized) return
        styleModeButton(lowStrengthButton, portraitStrength < 0.6f)
        styleModeButton(mediumStrengthButton, portraitStrength in 0.6f..0.8f)
        styleModeButton(highStrengthButton, portraitStrength > 0.8f)
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
                        val bitmap = decodeCapturedImage(output)
                        processAndSave(bitmap, selectedMode)
                    } catch (e: Exception) {
                        showFailure("Enhancement failed: ${e.message ?: e.javaClass.simpleName}")
                    } finally {
                        output.delete()
                    }
                }
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
            val result = enhanced
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
        modelExecutor.shutdownNow()
        maximEnhancer?.close()
        denoiser?.close()
        sceneSegmenter?.close()
        faceProtector?.close()
        super.onDestroy()
    }
}
