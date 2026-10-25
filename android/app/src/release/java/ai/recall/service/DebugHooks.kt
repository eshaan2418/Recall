package ai.recall.service

import android.content.Intent

/** Release builds: no file-replay input; the camera is the only frame source. */
object DebugHooks {
    fun replaySource(intent: Intent?): FrameSource? = null
}
