package ai.recall.service

import android.content.Intent
import android.graphics.Bitmap
import android.media.MediaMetadataRetriever
import java.io.File

/** Debug builds only: inject a recorded video file as camera input (e2e demo / instrumented tests). */
object DebugHooks {
    const val EXTRA_REPLAY_FILE = "replay_file"
    const val EXTRA_REPLAY_REALTIME = "replay_realtime"

    fun replaySource(intent: Intent?): FrameSource? {
        val path = intent?.getStringExtra(EXTRA_REPLAY_FILE) ?: return null
        return ReplaySource(File(path), realtime = intent.getBooleanExtra(EXTRA_REPLAY_REALTIME, false))
    }
}

/**
 * Recorded test video as camera input: decodes [file] at [rateHz] (2 fps, like the camera path), frames scaled to
 * 456×256, timestamps start at [startEpochS]. realtime=false: as fast as the pipeline consumes (blocking source).
 */
class ReplaySource(private val file: File, private val rateHz: Double = 2.0, private val realtime: Boolean = false,
                   private val startEpochS: Double = System.currentTimeMillis() / 1000.0) : FrameSource {
    @Volatile private var running = false
    private var thread: Thread? = null
    @Volatile var frames = 0
        private set
    override val blocking get() = !realtime

    override fun start(onFrame: (Bitmap, Double) -> Unit, onEnd: (error: String?) -> Unit) {
        running = true
        thread = Thread({
            var err: String? = null
            val r = MediaMetadataRetriever()
            try {
                r.setDataSource(file.absolutePath)
                val durMs = r.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION)!!.toLong()
                var tMs = 0.0
                val stepMs = 1000.0 / rateHz
                while (running && tMs < durMs) {
                    val bm = r.getFrameAtTime((tMs * 1000).toLong(), MediaMetadataRetriever.OPTION_CLOSEST)
                    if (bm != null) {
                        onFrame(Bitmap.createScaledBitmap(bm, 456, 256, true), startEpochS + tMs / 1000.0)
                        frames++
                    }
                    if (realtime) Thread.sleep(stepMs.toLong())
                    tMs += stepMs
                }
            } catch (e: Exception) {
                err = "replay failed: ${e.message}"
            } finally {
                r.release()
            }
            onEnd(err)
        }, "recall-replay").also { it.start() }
    }

    fun join() = thread?.join()

    override fun stop() { running = false }
}
