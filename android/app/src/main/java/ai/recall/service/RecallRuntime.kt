package ai.recall.service

import ai.recall.core.Bundle
import ai.recall.core.OperatingPoint
import ai.recall.core.QueryEngine
import ai.recall.data.MemoryDb
import ai.recall.ml.LiteRtEmbedder
import ai.recall.ml.LiteRtGate
import ai.recall.ml.LiteRtHandObject
import ai.recall.ml.LiteRtModel
import ai.recall.privacy.BlazeFaceDetector
import ai.recall.privacy.Redactor
import ai.recall.privacy.ScreenDetector
import ai.recall.privacy.SensitiveRegionDetector
import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow

/**
 * Process-wide singletons (models are expensive to load). Survive activity recreation (rotation); after process
 * death they are rebuilt lazily, and all memory/settings live in the encrypted DB, so nothing is lost.
 */
object RecallRuntime {
    private const val TAG = "Recall"
    val recording = MutableStateFlow(false)
    /** Human-readable status of the capture service (errors, fallbacks, backoff). */
    val status = MutableStateFlow("")
    /** Bumped whenever the memory changes (new events, forget, wipe) so the UI can refresh. */
    val memoryVersion = MutableStateFlow(0L)

    @Volatile private var db: MemoryDb? = null
    @Volatile private var bundle: Bundle? = null
    @Volatile private var pipeline: Pipeline? = null
    @Volatile var loadError: String? = null
        private set

    /** Test hook (instrumented tests): replaces model loading with a synthetic pipeline. */
    @Volatile var pipelineFactory: ((Context, OperatingPoint) -> Pipeline?)? = null

    val MODEL_ASSETS = listOf("gate.tflite", "embedder.tflite", "hoi_student.tflite")
    val BUNDLE_ASSETS = listOf("heads.bin", "vocab.json", "calibrator.json", "answer_config.json")
    const val FACE_ASSET = "face_detector.tflite"

    // Settings keys (stored in the encrypted DB).
    const val KEY_WANT_RECORDING = "want_recording"
    const val KEY_RETENTION_DAYS = "retention_days"
    const val KEY_OPERATING_POINT = "operating_point"
    const val DEFAULT_RETENTION_DAYS = 7

    fun missingAssets(ctx: Context): List<String> = (MODEL_ASSETS + BUNDLE_ASSETS).filter { !LiteRtModel.hasAsset(ctx, it) }

    fun missingBundle(ctx: Context): List<String> = BUNDLE_ASSETS.filter { !LiteRtModel.hasAsset(ctx, it) }

    @Synchronized fun db(ctx: Context): MemoryDb = db ?: MemoryDb(ctx.applicationContext).also { db = it }

    /** Query bundle (heads, vocab, calibrator). Null if not installed or invalid ([loadError] says why). */
    @Synchronized fun bundle(ctx: Context): Bundle? {
        bundle?.let { return it }
        val miss = missingBundle(ctx)
        if (miss.isNotEmpty()) { loadError = "missing ${miss.joinToString()}"; return null }
        val a = ctx.assets
        return try {
            Bundle.load(a.open("vocab.json").bufferedReader().readText(), a.open("heads.bin").readBytes(),
                a.open("calibrator.json").bufferedReader().readText(), a.open("answer_config.json").bufferedReader().readText())
                .also { bundle = it; loadError = null }
        } catch (e: Exception) {
            loadError = "invalid query bundle: ${e.message}"
            Log.e(TAG, "bundle load failed", e)
            null
        }
    }

    fun engine(ctx: Context): QueryEngine? = bundle(ctx)?.let { QueryEngine(it) }

    /** Face detector + screen heuristic; empty if the face model is missing → Redactor stores no thumbnails. */
    fun redactionDetectors(ctx: Context, faceThresh: Float): List<SensitiveRegionDetector> =
        if (LiteRtModel.hasAsset(ctx, FACE_ASSET)) {
            try { listOf(BlazeFaceDetector(ctx, faceThresh), ScreenDetector()) } catch (e: Exception) {
                Log.e(TAG, "face detector failed to load; thumbnails disabled", e); emptyList()
            }
        } else emptyList()

    @Synchronized fun pipeline(ctx: Context, op: OperatingPoint = OperatingPoint()): Pipeline? {
        pipeline?.let { it.operatingPoint = op; return it }
        pipelineFactory?.let { f -> return f(ctx, op).also { pipeline = it } }
        val b = bundle(ctx) ?: return null
        val missing = MODEL_ASSETS.filter { !LiteRtModel.hasAsset(ctx, it) }
        if (missing.isNotEmpty()) { loadError = "missing ${missing.joinToString()}"; return null }
        return try {
            val pre = b.preprocess
            val emb = LiteRtEmbedder(ctx, mean = pre.embedMean, std = pre.embedStd)
            require(emb.dim == b.dim) { "embedder.tflite outputs ${emb.dim}-d, heads.bin expects ${b.dim}-d" }
            Pipeline(LiteRtGate(ctx), LiteRtHandObject(ctx, presentThresh = pre.hoiPresentThresh), emb, b, db(ctx),
                Redactor(redactionDetectors(ctx, pre.faceThresh)), op).also { pipeline = it; loadError = null }
        } catch (e: Exception) {
            loadError = "model load failed: ${e.message}"
            Log.e(TAG, "pipeline load failed", e)
            null
        }
    }

    fun pipelineOrNull(): Pipeline? = pipeline

    /** Drop the cached pipeline (tests; after changing models). */
    @Synchronized fun resetPipeline() { pipeline = null }

    fun retentionDays(ctx: Context): Int = db(ctx).setting(KEY_RETENTION_DAYS)?.toIntOrNull() ?: DEFAULT_RETENTION_DAYS

    /** Auto-delete: events older than the retention period. 0 days = keep forever. Returns rows deleted. */
    fun applyRetention(ctx: Context, nowS: Double = System.currentTimeMillis() / 1000.0): Int {
        val d = retentionDays(ctx)
        if (d <= 0) return 0
        return db(ctx).applyRetention(nowS, d.toDouble()).also { if (it > 0) memoryChanged() }
    }

    fun memoryChanged() { memoryVersion.value = memoryVersion.value + 1 }
}
