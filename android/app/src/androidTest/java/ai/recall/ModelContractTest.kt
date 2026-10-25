package ai.recall

import ai.recall.ml.LiteRtEmbedder
import ai.recall.ml.LiteRtGate
import ai.recall.ml.LiteRtHandObject
import ai.recall.ml.LiteRtModel
import ai.recall.privacy.BlazeFaceDetector
import ai.recall.privacy.Pixels
import ai.recall.service.RecallRuntime
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.math.sqrt

/**
 * Validates whichever exported models are installed (models/litert → APK assets) against the asset contract by
 * loading them through the production wrappers and running one inference. Models that are absent are reported.
 */
@RunWith(AndroidJUnit4::class)
class ModelContractTest {
    private val ctx = Synthetic.target
    private fun has(n: String) = LiteRtModel.hasAsset(ctx, n).also { if (!it) Log.i(TAG, "$n not installed — skipped") }
    private fun frame(w: Int = 456, h: Int = 256) = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
        for (y in 0 until h) for (x in 0 until w) setPixel(x, y, Color.rgb(x * 255 / w, y * 255 / h, 128))
    }

    @Test fun gate() {
        val name = listOf("gate.tflite", "gate_test.tflite").firstOrNull { has(it) } ?: return
        LiteRtGate(ctx, name).use { g ->
            val s = (0 until 10).map { g.score(Bitmap.createScaledBitmap(frame(), 64, 36, true), FloatArray(7)) }
            Log.i(TAG, "$name OK: scores ${s.take(3)}…")
            assertTrue(s.all { it in 0f..1f })
        }
    }

    @Test fun embedder() {
        if (!has("embedder.tflite")) return
        val b = RecallRuntime.bundle(ctx)
        LiteRtEmbedder(ctx).use { e ->
            val v = e.embed(frame())
            val n = sqrt(v.sumOf { (it * it).toDouble() })
            Log.i(TAG, "embedder.tflite OK: dim ${e.dim}, |v|=$n, gpu=${e.usingGpu}")
            assertTrue(v.all { it.isFinite() })
            if (b != null) assertEquals("embedder dim must match heads.bin", b.dim, e.dim)
        }
    }

    @Test fun hoiStudent() {
        if (!has("hoi_student.tflite")) return
        LiteRtHandObject(ctx).use { h ->
            val o = h.detect(frame())
            Log.i(TAG, "hoi_student.tflite OK: $o")
            assertEquals(2, o.size)
            assertTrue(o.all { it.state in 0..4 && it.objBox!!.all { v -> v in 0f..1f } })
        }
    }

    @Test fun faceDetector() {
        if (!has(RecallRuntime.FACE_ASSET)) return
        BlazeFaceDetector(ctx).use { f ->
            val boxes = f.detect(Pixels.of(frame()))
            Log.i(TAG, "face_detector.tflite OK: ${boxes.size} boxes on a gradient")
        }
    }

    @Test fun bundle() {
        if (RecallRuntime.missingBundle(ctx).isNotEmpty()) { Log.i(TAG, "bundle not installed — skipped"); return }
        val b = RecallRuntime.bundle(ctx)
        assertTrue("bundle loads: ${RecallRuntime.loadError}", b != null)
        Log.i(TAG, "bundle OK: ${b!!.nounNames.size} nouns, ${b.locNames.size} locations, dim ${b.dim}")
    }

    companion object { const val TAG = "RecallModels" }
}
