package com.aicamera.pro

import android.content.Context
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.channels.FileChannel

/**
 * Maps uncompressed TFLite assets directly from the APK.
 * Avoids copying hundreds of MB into app storage on first AI use.
 */
internal object ModelAsset {
    fun map(context: Context, assetName: String): ByteBuffer {
        val descriptor = context.assets.openFd(assetName)
        descriptor.use { afd ->
            FileInputStream(afd.fileDescriptor).use { stream ->
                return stream.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    afd.startOffset,
                    afd.declaredLength
                )
            }
        }
    }
}
