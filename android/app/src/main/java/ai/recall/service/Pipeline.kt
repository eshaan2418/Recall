package ai.recall.service

import ai.recall.core.Bundle
import ai.recall.core.Event
import ai.recall.core.EventExtractor
import ai.recall.core.HandObs
import ai.recall.core.OperatingPoint
import ai.recall.core.PORTABLE
import ai.recall.core.STATIONARY
import ai.recall.core.SchedulePlan
import ai.recall.core.SideObs
import ai.recall.core.StepPolicy
import ai.recall.core.dot
import ai.recall.core.unit
import ai.recall.data.MemoryDb
import ai.recall.ml.Embedder
import ai.recall.ml.Gate
import ai.recall.ml.HandObjectDetector
import ai.recall.privacy.Redactor
import android.graphics.Bitmap
import org.json.JSONObject
import kotlin.math.exp

/** Counters exported to files/counters.json for the energy harness (recall/energy/harness.py). */
data class PipelineStats(
    @Volatile var steps: Long = 0,
    @Volatile var gated: Long = 0,
    @Volatile var heavy: Long = 0,
    @Volatile var events: Long = 0,
    @Volatile var skippedBlockedPlace: Long = 0,
    @Volatile var droppedBlockedObject: Long = 0,
    @Volatile var thumbsStored: Long = 0,
    @Volatile var thumbsWithheld: Long = 0,
    @Volatile var embedCalls: Long = 0,
    @Volatile var lastPlan: String = "nominal",
    @Volatile var backoffLevel: Int = 0,
) {
    fun toJson(extra: Map<String, Any> = emptyMap()): JSONObject = JSONObject().apply {
        put("steps", steps); put("gated", gated); put("heavy", heavy); put("events", events)
        put("skipped_blocked_place", skippedBlockedPlace); put("dropped_blocked_object", droppedBlockedObject)
        put("thumbs_stored", thumbsStored); put("thumbs_withheld", thumbsWithheld); put("embed_calls", embedCalls)
        put("plan", lastPlan); put("backoff_level", backoffLevel)
        for ((k, v) in extra) put(k, v)
    }
}

/**
 * Per-step memory pipeline (2 steps/s): trigger policy (gate → burst controller, or all / uniform-k) → HOI student +
 * embedder → event extractor → privacy filters (blocked place / object, redaction) → encrypted store.
 * Must be driven from a single thread (CaptureService's executor); [flush] / [reloadPolicies] are safe to call from it.
 */
class Pipeline(
    private val gate: Gate,
    private val hoi: HandObjectDetector,
    private val embedder: Embedder,
    private val bundle: Bundle,
    private val db: MemoryDb,
    private val redactor: Redactor,
    op: OperatingPoint = OperatingPoint(),
    private val placeSim: Float = 0.80f,
) {
    val stats = PipelineStats()
    private val policy = StepPolicy(op, bundle.gateThreshold)
    private var extractor = EventExtractor()
    private var step = 0L
    private var prevBoxes: Array<FloatArray?> = arrayOf(null, null)
    private var prevProcessed = false
    private val places = db.places().map { it.id to it.centroid }.toMutableList()
    private val placeCounts = db.places().map { it.n }.toMutableList()
    @Volatile private var blockedPlaces = db.places().filter { it.blocked }.map { it.id }.toSet()
    @Volatile private var blockedObjects = db.blockedObjects()
    private var lastFrame: Bitmap? = null

    /** Place cluster of the most recent heavy step (for "never record here"); -1 if unknown. */
    @Volatile var currentPlace: Int = -1
        private set

    var operatingPoint: OperatingPoint
        get() = policy.op
        set(v) { policy.op = v }

    fun reloadPolicies() {
        blockedPlaces = db.places().filter { it.blocked }.map { it.id }.toSet()
        blockedObjects = db.blockedObjects()
    }

    fun onFrame(frame: Bitmap, tEpochS: Double, imu: FloatArray, plan: SchedulePlan) {
        stats.steps++
        policy.applyPlan(plan)
        stats.lastPlan = policy.plan.reason
        stats.backoffLevel = policy.plan.level
        val s = step++
        val d = policy.decide(s) { stats.gated++; gate.score(Bitmap.createScaledBitmap(frame, 64, 36, true), imu) }
        if (!d.heavy) { prevProcessed = false; return }
        stats.heavy++
        val scene = embed(frame)
        val pid = assignPlace(scene)
        currentPlace = pid
        if (pid in blockedPlaces) {
            stats.skippedBlockedPlace++; prevProcessed = false
            extractor = EventExtractor() // drop any open bout: nothing from a blocked place is remembered
            return
        }
        val hands: Array<HandObs> = hoi.detect(frame)
        val sides = Array(2) { side ->
            val h = hands[side]
            val contact = h.present && (h.state == PORTABLE || h.state == STATIONARY) && h.objBox != null
            val objEmb = if (contact) embed(crop(frame, h.objBox!!, 1.15f)) else null
            val pb = prevBoxes[side]
            val rel = if (prevProcessed && pb != null) embed(crop(frame, pb, 3f)) else null
            prevBoxes[side] = if (contact) h.objBox else null
            SideObs(h, objEmb, rel)
        }
        prevProcessed = true
        lastFrame = frame
        for (e in extractor.onStep(s.toInt(), tEpochS, sides, scene)) store(e, frame, pid)
    }

    fun flush() {
        lastFrame?.let { f -> for (e in extractor.finish()) store(e, f, currentPlace) }
        extractor = EventExtractor()
        lastFrame = null
        prevProcessed = false
    }

    private fun store(e: Event, frame: Bitmap, pid: Int) {
        val op = bundle.obj.logProbs(e.objEmb).map { exp(it) }
        val objTop = op.indices.sortedByDescending { op[it] }.take(5).map { it to op[it].toFloat() }
        if (objTop.first().first in blockedObjects) { stats.droppedBlockedObject++; return } // "never track this object"
        val lp = bundle.loc.logProbs(e.locEmb).map { exp(it) }
        val locTop = lp.indices.sortedByDescending { lp[it] }.take(5).map { it to lp[it].toFloat() }
        e.thumb = redactor.thumbnail(crop(frame, e.box, 3f))
        if (e.thumb != null) stats.thumbsStored++ else stats.thumbsWithheld++
        db.insert(e, objTop, locTop, pid)
        stats.events++
    }

    private fun embed(b: Bitmap): FloatArray { stats.embedCalls++; return unit(embedder.embed(b)) }

    private fun assignPlace(scene: FloatArray): Int {
        var best = -1; var bs = -1f
        for ((k, p) in places.withIndex()) { val sim = dot(p.second, scene); if (sim > bs) { bs = sim; best = k } }
        if (best >= 0 && bs >= placeSim) {
            val n = ++placeCounts[best]
            val c = places[best].second
            val upd = unit(FloatArray(c.size) { c[it] + (scene[it] - c[it]) / n })
            places[best] = places[best].first to upd
            if (n % 20 == 0) db.upsertPlace(places[best].first, upd, n)
            return places[best].first
        }
        val id = (places.maxOfOrNull { it.first } ?: -1) + 1
        places += id to scene; placeCounts += 1
        db.upsertPlace(id, scene, 1)
        return id
    }

    companion object {
        /** Crop around normalised [box] scaled by [f] (recall/perception/extract.py::expand_box + crop). */
        fun crop(bm: Bitmap, box: FloatArray, f: Float, minSize: Float = 0.06f): Bitmap {
            val r = cropRect(bm.width, bm.height, box, f, minSize)
            return Bitmap.createBitmap(bm, r[0], r[1], r[2] - r[0], r[3] - r[1])
        }

        /** Pixel rect [x0, y0, x1, y1] (pure; unit-tested). */
        fun cropRect(w: Int, h: Int, box: FloatArray, f: Float, minSize: Float = 0.06f): IntArray {
            val cx = (box[0] + box[2]) / 2; val cy = (box[1] + box[3]) / 2
            val bw = maxOf((box[2] - box[0]) * f, minSize); val bh = maxOf((box[3] - box[1]) * f, minSize)
            val l = maxOf(0f, cx - bw / 2); val t = maxOf(0f, cy - bh / 2)
            val r = minOf(1f, cx + bw / 2); val b = minOf(1f, cy + bh / 2)
            val x0 = (l * w).toInt().coerceIn(0, w - 2); val y0 = (t * h).toInt().coerceIn(0, h - 2)
            val x1 = maxOf(x0 + 2, Math.ceil((r * w).toDouble()).toInt()).coerceAtMost(w)
            val y1 = maxOf(y0 + 2, Math.ceil((b * h).toDouble()).toInt()).coerceAtMost(h)
            return intArrayOf(x0, y0, x1, y1)
        }
    }
}
