package ai.recall

import ai.recall.core.AnswerConfig
import ai.recall.core.BlendHead
import ai.recall.core.Bundle
import ai.recall.core.Calibrator
import ai.recall.core.HandObs
import ai.recall.core.OperatingPoint
import ai.recall.core.PORTABLE
import ai.recall.data.MemoryDb
import ai.recall.ml.Embedder
import ai.recall.ml.Gate
import ai.recall.ml.HandObjectDetector
import ai.recall.privacy.Box
import ai.recall.privacy.Pixels
import ai.recall.privacy.Redactor
import ai.recall.privacy.SensitiveRegionDetector
import ai.recall.service.Pipeline
import android.content.Context
import android.graphics.Bitmap
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import kotlin.math.abs
import kotlin.math.sqrt

/**
 * Synthetic stand-ins used when the real LiteRT models are not installed ("plumbing" mode). They exercise the full
 * device pipeline (video decode → trigger → perception interfaces → event extractor → redaction → SQLCipher → query
 * engine → calibrated grounded answer) but NOT model accuracy. Colours match android/tools/make_test_video.py.
 */
object Synthetic {
    val PROTOS = arrayOf(
        intArrayOf(255, 215, 0),   // 0 keys (yellow)
        intArrayOf(120, 72, 30),   // 1 table (brown)
        intArrayOf(150, 150, 150), // 2 counter (grey)
        intArrayOf(40, 90, 200),   // 3 sink (blue)
        intArrayOf(224, 172, 140), // 4 skin
        intArrayOf(235, 225, 205), // 5 wall
        intArrayOf(30, 30, 30),    // 6 dark (phone)
        intArrayOf(250, 250, 250), // 7 white (mug)
    )

    /** Fires on every step (synthetic clips have no learned-gate signal worth testing). */
    class AlwaysOnGate : Gate {
        var calls = 0
        override fun score(frame36x64: Bitmap, imu: FloatArray): Float { calls++; return 1f }
    }

    /** 8-d colour histogram over [PROTOS] (nearest prototype within distance 90), L2-normalised. */
    class ColourEmbedder : Embedder {
        override val dim = PROTOS.size
        override fun embed(img: Bitmap): FloatArray {
            val s = Bitmap.createScaledBitmap(img, 48, 48, false)
            val px = IntArray(48 * 48).also { s.getPixels(it, 0, 48, 0, 0, 48, 48) }
            val h = FloatArray(dim)
            for (p in px) {
                val r = (p shr 16) and 0xff; val g = (p shr 8) and 0xff; val b = p and 0xff
                var best = -1; var bd = Int.MAX_VALUE
                for ((k, c) in PROTOS.withIndex()) {
                    val d = (r - c[0]) * (r - c[0]) + (g - c[1]) * (g - c[1]) + (b - c[2]) * (b - c[2])
                    if (d < bd) { bd = d; best = k }
                }
                if (bd < 90 * 90) h[best] += 1f
            }
            val n = sqrt(h.sumOf { (it * it).toDouble() }).toFloat().coerceAtLeast(1e-6f)
            return FloatArray(dim) { h[it] / n }
        }
    }

    /**
     * Colour-key hand–object "detector": skin pixels = right hand, yellow pixels = object; contact (PORTABLE) if the
     * hand box (grown by [margin] px) overlaps the object box. Debug/test only — the HOI student cannot see drawn hands.
     */
    class ColourKeyHandObject(private val margin: Int = 6, private val minPx: Int = 40) : HandObjectDetector {
        private fun near(p: Int, c: IntArray, tol: Int) =
            abs(((p shr 16) and 0xff) - c[0]) < tol && abs(((p shr 8) and 0xff) - c[1]) < tol && abs((p and 0xff) - c[2]) < tol

        private fun bbox(px: IntArray, w: Int, h: Int, test: (Int) -> Boolean): IntArray? {
            var l = w; var t = h; var r = -1; var b = -1; var n = 0
            for (y in 0 until h) for (x in 0 until w) if (test(px[y * w + x])) { n++; l = minOf(l, x); t = minOf(t, y); r = maxOf(r, x); b = maxOf(b, y) }
            return if (n >= minPx) intArrayOf(l, t, r + 1, b + 1) else null
        }

        override fun detect(frame: Bitmap): Array<HandObs> {
            val w = frame.width; val h = frame.height
            val px = IntArray(w * h).also { frame.getPixels(it, 0, w, 0, 0, w, h) }
            val hand = bbox(px, w, h) { near(it, PROTOS[4], 32) }
            val obj = bbox(px, w, h) { p -> ((p shr 16) and 0xff) > 190 && ((p shr 8) and 0xff) > 160 && (p and 0xff) < 90 }
            val none = HandObs(false, 0f, 0, null)
            if (hand == null) return arrayOf(none, none)
            val contact = obj != null && hand[0] - margin < obj[2] && obj[0] < hand[2] + margin &&
                hand[1] - margin < obj[3] && obj[1] < hand[3] + margin
            val box = obj?.let { floatArrayOf(it[0] / w.toFloat(), it[1] / h.toFloat(), it[2] / w.toFloat(), it[3] / h.toFloat()) }
            return arrayOf(none, HandObs(true, 0.95f, if (contact) PORTABLE else 0, if (contact) box else null))
        }
    }

    /** Redaction "detector" that flags nothing (no faces in synthetic clips) so thumbnails are stored. */
    class NoFaces : SensitiveRegionDetector {
        override val name = "face"
        override fun detect(img: Pixels): List<Box> = emptyList()
    }

    private fun onehot(k: Int) = FloatArray(PROTOS.size).also { it[k] = 1f }

    val NOUNS = listOf("keys", "phone", "mug")
    val LOCS = listOf("table", "counter", "sink")

    fun bundle(): Bundle {
        val obj = BlendHead(arrayOf(onehot(0), onehot(6), onehot(7)), 20f, 1f)
        val loc = BlendHead(arrayOf(onehot(1), onehot(2), onehot(3)), 20f, 1f)
        val cal = Calibrator(listOf("score", "loc_p"), doubleArrayOf(3.0, 1.0), -1.0, DoubleArray(0), DoubleArray(0), 0.2)
        val cfg = AnswerConfig()
        return Bundle(NOUNS, NOUNS.withIndex().associate { it.value to it.index } + mapOf("key" to 0), listOf(0, 1, 2), LOCS, obj, loc, cal,
            mapOf("where" to cfg, "temporal" to cfg, "state" to cfg), 0.5f, emptyMap())
    }

    fun pipeline(db: MemoryDb, op: OperatingPoint = OperatingPoint(), bundle: Bundle = bundle()) =
        Pipeline(AlwaysOnGate(), ColourKeyHandObject(), ColourEmbedder(), bundle, db, Redactor(listOf(NoFaces())), op)

    val target: Context get() = InstrumentationRegistry.getInstrumentation().targetContext

    /** Copies the synthetic clip from the test APK's assets into the app's private files dir. */
    fun clip(name: String = "keys_on_table.webm"): File {
        val out = File(target.filesDir, "e2e/$name").apply { parentFile!!.mkdirs() }
        InstrumentationRegistry.getInstrumentation().context.assets.open("e2e/$name").use { i -> out.outputStream().use { i.copyTo(it) } }
        return out
    }
}
