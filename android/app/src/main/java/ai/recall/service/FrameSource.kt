package ai.recall.service

import android.content.Context
import android.graphics.Bitmap
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import java.util.concurrent.Executor

/** Delivers frames at ~rateHz to onFrame (frame, timestamp epoch seconds); onEnd when the source is exhausted/failed. */
interface FrameSource {
    /** True if onFrame must not drop frames (file replay: deterministic, as fast as processing allows). */
    val blocking: Boolean get() = false

    fun start(onFrame: (Bitmap, Double) -> Unit, onEnd: (error: String?) -> Unit)
    fun stop()
}

/** Back camera via CameraX ImageAnalysis (keep-only-latest), throttled to rateHz, frames scaled to 456×256. */
class CameraXSource(private val ctx: Context, private val owner: LifecycleOwner, private val analyzerExec: Executor,
                    private val rateHz: Double = 2.0) : FrameSource {
    private var provider: ProcessCameraProvider? = null
    private var lastNs = 0L

    override fun start(onFrame: (Bitmap, Double) -> Unit, onEnd: (error: String?) -> Unit) {
        val f = ProcessCameraProvider.getInstance(ctx)
        f.addListener({
            try {
                val p = f.get().also { provider = it }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .build()
                analysis.setAnalyzer(analyzerExec) { img ->
                    try {
                        val now = System.nanoTime()
                        if (now - lastNs >= (1e9 / rateHz).toLong()) {
                            lastNs = now
                            val bm = img.toBitmap()
                            val small = Bitmap.createScaledBitmap(bm, 456, 256, true) // same resolution class as the EPIC frames
                            onFrame(small, System.currentTimeMillis() / 1000.0)
                        }
                    } finally {
                        img.close()
                    }
                }
                p.unbindAll()
                p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
            } catch (e: Exception) { // no camera, camera in use, permission revoked …
                onEnd("camera unavailable: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(ctx))
    }

    override fun stop() {
        ContextCompat.getMainExecutor(ctx).execute { provider?.unbindAll() }
    }
}
